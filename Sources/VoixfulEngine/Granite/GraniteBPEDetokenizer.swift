// Granite byte-level BPE detokenizer.
//
// The convert pipeline precomputes each token id's RAW BYTES (GPT-2
// byte<->unicode table applied at convert time) and ships them as
// `tokens.json` inside the .aimodel dir: {"format":1, "tokens_b64":[...]}.
// Detokenization is then pure concatenation — no merges, no vocab logic —
// and UTF-8 multi-byte sequences that span token boundaries assemble
// correctly because the join happens at the byte level.

import Foundation

struct GraniteBPEDetokenizer {
    private let tokenBytes: [Data]

    init(tokensURL: URL) throws {
        let data = try Data(contentsOf: tokensURL)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              (obj["format"] as? Int) == 1,
              let table = obj["tokens_b64"] as? [String]
        else {
            throw ASRError.modelLoadFailed("malformed tokens.json at \(tokensURL.path)")
        }
        self.tokenBytes = table.map { Data(base64Encoded: $0) ?? Data() }
    }

    var count: Int { tokenBytes.count }

    func decode(_ ids: [Int32]) -> String {
        var bytes = Data()
        for id in ids where id >= 0 && Int(id) < tokenBytes.count {
            bytes.append(tokenBytes[Int(id)])
        }
        return String(data: bytes, encoding: .utf8) ?? ""
    }
}
