#!/usr/bin/env python3
"""CPU-only checks of kernel_filter.py on synthetic preprocessed Metal text (no Metal compiler).

    python3 -I tests/test_kernel_filter.py
"""
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "metallib_slim"
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("kernel_filter", LIB / "kernel_filter.py")
K = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = K  # @dataclass + `from __future__ import annotations` look the module up in sys.modules (Python 3.9)
_spec.loader.exec_module(K)

BODY = """#pragma METAL internals : enable
constant bool has_w [[function_constant(0)]];
template <typename T, typename U, typename Op>
[[kernel]] void unary_v(device const T* in, device U* out, uint index [[thread_position_in_grid]]) {
  // template [[host_name("fake")]] [[kernel]] decltype(x) x;  (a comment, not a statement)
  const char* s = "[[host_name(\\"fake2\\")]]";
  out[index] = Op()(in[index]);
}
"""
INST = [
    'template [[host_name( "vn_" "FromFP8" "uint8" "float32")]] [[kernel]] decltype(unary_v<uint8_t, float, FromFP8>) unary_v<uint8_t, float, FromFP8>;\n',
    'template [[host_name("vn_Abs" "float32")]] [[kernel]] decltype(unary_v<float, float, Abs>) unary_v<float, float, Abs>;\n',
    'template [[host_name("vn_Abs" "float16")]]\n  [[kernel]] decltype(unary_v<half, half, Abs>)\n  unary_v<half, half, Abs>;\n',
]
TEXT = BODY + "".join(INST) + "constant int tail = 1;\n"
NAMES = ["vn_FromFP8uint8float32", "vn_Absfloat32", "vn_Absfloat16"]


class ScanTests(unittest.TestCase):
    def test_finds_every_instantiation_with_concatenated_literals_and_ignores_comments_and_strings(self):
        found = K.scan(TEXT)
        self.assertEqual([i.name for i in found], NAMES)
        self.assertEqual(found[0].literals, ("vn_", "FromFP8", "uint8", "float32"))
        self.assertTrue(found[2].declaration.strip().startswith("decltype(unary_v<half, half, Abs>)"))
        self.assertTrue(TEXT[found[1].start:found[1].end].startswith("template [[host_name"))
        self.assertTrue(TEXT[found[1].start:found[1].end].endswith(";"))

    def test_text_without_host_name_scans_empty(self):
        self.assertEqual(K.scan("void f() {}\n"), [])

    def test_malformed_or_unknown_syntax_fails_closed(self):
        bad = {
            "host_name on a definition": '[[host_name("x")]] [[kernel]] void f() {}\n',
            "no template": 'int a;\nfoo [[host_name("x")]] [[kernel]] decltype(f) f;\n',
            "no kernel attribute": 'template [[host_name("x")]] decltype(f<int>) f<int>;\n',
            "unterminated": 'template [[host_name("x")]] [[kernel]] decltype(f<int>) f<int>\n',
            "brace in declaration": 'template [[host_name("x")]] [[kernel]] decltype(f<int>) f<int> {}\n',
            "non-literal name": 'template [[host_name(NAME "x")]] [[kernel]] decltype(f<int>) f<int>;\n',
            "empty name": 'template [[host_name()]] [[kernel]] decltype(f<int>) f<int>;\n',
            "escape in name": 'template [[host_name("a\\\\b")]] [[kernel]] decltype(f<int>) f<int>;\n',
            "non-identifier name": 'template [[host_name("a<b")]] [[kernel]] decltype(f<int>) f<int>;\n',
            "duplicate": 'template [[host_name("x")]] [[kernel]] decltype(f<int>) f<int>;\ntemplate [[host_name("x")]] [[kernel]] decltype(f<float>) f<float>;\n',
            "unbalanced paren": 'template [[host_name("x")]] [[kernel]] decltype(f<int>)) f<int>;\n',
            "unterminated comment": '/* never closed\ntemplate [[host_name("x")]] [[kernel]] decltype(f<int>) f<int>;\n',
            "template glued to an identifier": 'mytemplate [[host_name("x")]] [[kernel]] decltype(f<int>) f<int>;\n',
            "bare host_name word": 'int host_name = 1;\n',
        }
        for label, text in bad.items():
            with self.subTest(label):
                with self.assertRaises(K.FilterError):
                    K.scan(text)

    def test_statement_may_follow_a_directive_or_a_closing_brace(self):
        text = '#pragma x\ntemplate [[host_name("a")]] [[kernel]] decltype(f<int>) f<int>;\nvoid g() {}\ntemplate [[host_name("b")]] [[kernel]] decltype(f<long>) f<long>;\n'
        self.assertEqual([i.name for i in K.scan(text)], ["a", "b"])


class FilterTests(unittest.TestCase):
    def test_removes_only_unselected_statements_and_keeps_every_other_byte(self):
        filtered, report = K.filter_text(TEXT, {"vn_Absfloat32"})
        # a statement ends at its `;`: the newline after it is not part of the removed range
        self.assertEqual(filtered, BODY + "\n" + INST[1] + "\n" + "constant int tail = 1;\n")
        self.assertEqual((report["instantiations"], report["kept"], report["removed"]), (3, 1, 2))
        self.assertEqual(report["keptNames"], ["vn_Absfloat32"])
        self.assertTrue(report["otherTextIdentical"])
        self.assertIn("out[index] = Op()(in[index]);", filtered)  # the kernel body is untouched
        self.assertIn("function_constant(0)", filtered)
        self.assertEqual(report["removedBytes"], len((INST[0].rstrip("\n") + INST[2].rstrip("\n")).encode()))

    def test_keeping_everything_is_the_identity_and_keeping_nothing_leaves_no_instantiation(self):
        same, _ = K.filter_text(TEXT, set(NAMES))
        self.assertEqual(same, TEXT)
        none, report = K.filter_text(TEXT, set())
        self.assertEqual(none, BODY + "\n\n\n" + "constant int tail = 1;\n")
        self.assertEqual(K.scan(none), [])
        self.assertEqual(report["removed"], 3)

    def test_names_that_are_not_in_the_file_are_ignored_by_the_file_level_filter(self):
        filtered, report = K.filter_text(TEXT, {"vn_Absfloat32", "no_such_kernel"})
        self.assertEqual(report["kept"], 1)
        self.assertIn(INST[1], filtered)

    def test_prove_rejects_text_that_differs_beyond_the_removed_statements(self):
        found = K.scan(TEXT)
        filtered, _ = K.filter_text(TEXT, {"vn_Absfloat32"})
        with self.assertRaises(K.FilterError):
            K.prove(TEXT, filtered.replace("tail = 1", "tail = 2"), found, {"vn_Absfloat32"})
        with self.assertRaises(K.FilterError):
            K.prove(TEXT, filtered, found, {"vn_Absfloat32", "vn_Absfloat16"})  # claims a different kept set


if __name__ == "__main__":
    unittest.main()
