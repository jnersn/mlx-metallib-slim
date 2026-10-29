#!/usr/bin/env python3
"""Removes UNSELECTED explicit kernel instantiations from preprocessed Metal text. Fails closed.

Input is the output of `xcrun metal -E -P` for one MLX kernel file, so every instantiate_kernel(...) macro is already expanded to
    template [[host_name("a" "b" "c")]] [[kernel]] decltype(f<T>) f<T>;
This module finds exactly those statements and nothing else:
  * `template` (at a statement start), `[[host_name( "lit" ... )]]` (adjacent string literals only, concatenated), `[[kernel]]`, a declaration that
    ends at the first `;` outside parentheses and holds no brace;
  * any other use of `host_name`, an unterminated statement, a brace in the declaration, a non-literal in the name, a backslash in a name, an empty or
    non-identifier name, a duplicate name: FilterError (no regex fallback, no guess).
`filter_text` deletes ONLY the byte ranges of unselected instantiations. Every kernel body, helper, header, function constant and kept instantiation
stays byte for byte; `prove` recomputes that from scratch (the rest of the text is exactly the complement of the removed ranges, the filtered text
rescans to exactly the kept names in order, every removed range is one whole instantiation statement of an unselected name).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


class FilterError(Exception):
    pass


@dataclass(frozen=True)
class Instantiation:
    start: int  # offset of `template`
    end: int  # offset one past the terminating `;`
    name: str  # concatenated host_name literals
    literals: tuple
    declaration: str  # text between `[[kernel]]` and `;`


HOST = re.compile(r"\[\[\s*host_name\s*\(")
KERNEL = re.compile(r"\s*\[\[\s*kernel\s*\]\]")
CLOSE = re.compile(r"\s*\]\]")
NAME = re.compile(r"^[A-Za-z0-9_]+$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mask_code(text: str) -> str:
    """Same length as `text`; comment characters become spaces and string/char literal contents become `_` (quotes and newlines kept)."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise FilterError(f"unterminated comment at offset {i}")
            j += 2
            for k in range(i, j):
                if text[k] != "\n":
                    out[k] = " "
            i = j
        elif c == '"' or (c == "'" and (i == 0 or not text[i - 1].isalnum())):
            j = i + 1
            while True:
                if j >= n or text[j] == "\n":
                    raise FilterError(f"unterminated literal at offset {i}")
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == c:
                    break
                j += 1
            for k in range(i + 1, j):
                out[k] = "_"
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _skip_space(mask: str, i: int) -> int:
    while i < len(mask) and mask[i].isspace():
        i += 1
    return i


def _parse_one(text: str, mask: str, match: re.Match) -> Instantiation:
    where = f"host_name at offset {match.start()}"
    # backwards: `template` at a statement start
    j = match.start() - 1
    while j >= 0 and mask[j].isspace():
        j -= 1
    if j < 7 or mask[j - 7:j + 1] != "template" or (j - 8 >= 0 and (mask[j - 8].isalnum() or mask[j - 8] == "_")):
        raise FilterError(f"{where}: not preceded by `template` (host_name is only accepted on explicit instantiations)")
    start = j - 7
    p = start - 1
    while p >= 0 and mask[p].isspace():
        p -= 1
    if p >= 0 and mask[p] not in ";}":
        line_end = text.find("\n", p)
        line = text[text.rfind("\n", 0, p) + 1:len(text) if line_end < 0 else line_end]
        if not line.lstrip().startswith("#"):
            raise FilterError(f"{where}: `template` does not start a statement (previous text ends with {text[max(0, p - 20):p + 1]!r})")
    # forwards: string literals, `)`, `]]`, `[[kernel]]`
    pos = match.end()
    literals = []
    while True:
        pos = _skip_space(mask, pos)
        if pos >= len(mask):
            raise FilterError(f"{where}: unterminated host_name")
        if mask[pos] == '"':
            close = mask.find('"', pos + 1)
            if close < 0:
                raise FilterError(f"{where}: unterminated string")
            literal = text[pos + 1:close]
            if "\\" in literal or "\n" in literal:
                raise FilterError(f"{where}: escape or newline in a name literal {literal!r}")
            literals.append(literal)
            pos = close + 1
        elif mask[pos] == ")":
            pos += 1
            break
        else:
            raise FilterError(f"{where}: unexpected {mask[pos]!r} in the host_name arguments (only adjacent string literals are accepted)")
    if not literals:
        raise FilterError(f"{where}: no name literal")
    name = "".join(literals)
    if not NAME.match(name):
        raise FilterError(f"{where}: name {name!r} is not an identifier")
    closing = CLOSE.match(mask, pos)
    if not closing:
        raise FilterError(f"{where}: `]]` missing after host_name({name})")
    kernel = KERNEL.match(mask, closing.end())
    if not kernel:
        raise FilterError(f"{where}: `[[kernel]]` does not follow host_name({name})")
    decl_start = kernel.end()
    depth, i = 0, decl_start
    while True:
        if i >= len(mask):
            raise FilterError(f"{where}: no `;` ends the declaration of {name}")
        c = mask[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                raise FilterError(f"{where}: unbalanced `)` in the declaration of {name}")
        elif c in "{}":
            raise FilterError(f"{where}: {c!r} in the declaration of {name} (this is not a pure explicit instantiation)")
        elif c == ";" and depth == 0:
            break
        i += 1
    return Instantiation(start, i + 1, name, tuple(literals), text[decl_start:i])


def scan(text: str) -> list:
    """Every explicit instantiation with a host_name, in order. Every occurrence of host_name must be one (otherwise FilterError)."""
    mask = mask_code(text)
    found = [_parse_one(text, mask, m) for m in HOST.finditer(mask)]
    for earlier, later in zip(found, found[1:]):
        if later.start < earlier.end:
            raise FilterError(f"{later.name}: overlaps {earlier.name}")
    names = [item.name for item in found]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise FilterError(f"duplicate host_name(s) in one file: {duplicates[:5]}")
    if len(re.findall(r"\bhost_name\b", mask)) != len(found):
        raise FilterError("host_name occurs outside the accepted form")
    return found


def filter_text(text: str, keep) -> tuple:
    """(filtered text, report). Removes exactly the instantiations whose name is not in `keep`."""
    keep = set(keep)
    found = scan(text)
    removed = [item for item in found if item.name not in keep]
    pieces, cursor = [], 0
    for item in removed:
        pieces.append(text[cursor:item.start])
        cursor = item.end
    pieces.append(text[cursor:])
    filtered = "".join(pieces)
    report = prove(text, filtered, found, keep)
    return filtered, report


def prove(original: str, filtered: str, found: list, keep) -> dict:
    """Independent check of a filter result; raises FilterError on any deviation."""
    keep = set(keep)
    removed = [item for item in found if item.name not in keep]
    kept = [item for item in found if item.name in keep]
    gaps, cursor = [], 0
    for item in removed:
        if not (original[item.start:item.end].startswith("template") and original[item.start:item.end].endswith(";")):
            raise FilterError(f"removed range of {item.name} is not a whole statement")
        gaps.append(original[cursor:item.start])
        cursor = item.end
    gaps.append(original[cursor:])
    if "".join(gaps) != filtered:
        raise FilterError("the filtered text is not the original minus the removed instantiation statements")
    if [i.name for i in scan(filtered)] != [i.name for i in kept]:
        raise FilterError("the filtered text does not rescan to exactly the kept instantiations")
    removed_bytes = sum(len(original[i.start:i.end].encode("utf-8")) for i in removed)
    if len(original.encode("utf-8")) - removed_bytes != len(filtered.encode("utf-8")):
        raise FilterError("byte accounting of the removed statements does not match")
    return {"instantiations": len(found), "kept": len(kept), "removed": len(removed), "keptNames": [i.name for i in kept],
            "removedBytes": removed_bytes, "originalSHA256": sha256_text(original), "filteredSHA256": sha256_text(filtered),
            "otherTextIdentical": True}
