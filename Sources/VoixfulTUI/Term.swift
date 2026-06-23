// VoixfulTUI — tiny ANSI + terminal helpers.
//
// Model-agnostic: the live mic UI is shared by every Voixful model backend, so
// it lives here rather than in any one model's CLI.

import Foundation
#if canImport(Darwin)
import Darwin
#endif

enum Term {
    /// Only style when stdout is a real terminal; piped output stays plain.
    static let isTTY = isatty(STDOUT_FILENO) != 0

    /// Current terminal width (columns), best-effort.
    static var cols: Int {
        var ws = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0, ws.ws_col > 0 { return Int(ws.ws_col) }
        if let c = ProcessInfo.processInfo.environment["COLUMNS"], let n = Int(c), n > 0 { return n }
        return 80
    }

    /// Current terminal height (rows), best-effort.
    static var rows: Int {
        var ws = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0, ws.ws_row > 0 { return Int(ws.ws_row) }
        if let r = ProcessInfo.processInfo.environment["LINES"], let n = Int(r), n > 0 { return n }
        return 24
    }

    static let reset = "\u{1B}[0m"
    static let bold = "\u{1B}[1m"
    static let dim = "\u{1B}[2m"
    static let italic = "\u{1B}[3m"
    static func fg(_ c: Int) -> String { "\u{1B}[38;5;\(c)m" }

    /// "voixful" rendered as a teal→green 256-color gradient. Visible width 7.
    static func brand() -> String {
        let palette = [45, 44, 43, 79, 115, 151, 187]
        var s = bold
        for (i, ch) in Array("voixful").enumerated() { s += fg(palette[i]) + String(ch) }
        return s + reset
    }
}
