import CryptoKit
import Foundation
import Metal

// Reflects the function names of a compiled Metal library: the authority for "this export is PRESENT".
//
//   swiftc -O metallib_names.swift -o metallib-names
//   metallib-names <file.metallib> <out.json>
//
// Writes {path, bytes, sha256, functionCount, names[sorted]}. Needs a Metal device (it loads the library, it does not parse bytes).

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data("metallib-names: \(message)\n".utf8))
    exit(code)
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else { fail("usage: metallib-names <file.metallib> <out.json>", code: 2) }
let url = URL(fileURLWithPath: arguments[1])
guard let data = try? Data(contentsOf: url) else { fail("cannot read \(url.path)") }
guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
let library: MTLLibrary
do { library = try device.makeLibrary(URL: url) } catch { fail("cannot load \(url.path): \(error)") }
let names = library.functionNames.sorted()
let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
let object: [String: Any] = ["path": url.path, "bytes": data.count, "sha256": digest, "functionCount": names.count, "names": names]
do {
    let json = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: URL(fileURLWithPath: arguments[2]))
} catch { fail("cannot write \(arguments[2]): \(error)") }
print("\(names.count) functions, \(data.count) bytes, sha256 \(digest)")
