// VoixfulTUI — the live mic terminal display.
//
// Owns the terminal during a mic session: the transcription flows and wraps in
// the main scroll area (visible / scrollable), while a fixed footer is pinned
// at the bottom via a DEC scroll region:
//   blank spacer
//   ──────────── horizontal rule
//   voixful brand + colored dB level meter + numeric peak
//   input device · rate · channels   ·   Ctrl+C to stop
// All writes funnel through this actor, so the footer and the transcript never
// fight over the cursor. Falls back to plain line output when stdout is not a
// terminal or is too short for a pinned footer.

import Foundation
#if canImport(Darwin)
import Darwin
#endif

actor MicDisplay {
    private let tty: Bool
    private var rows = 0
    private var enabled = false          // pinned-footer mode active
    private var model = ""               // backend display name, e.g. "Nemotron"
    private var device = ""
    private var sr = 0
    private var ch = 0
    private var liveRows = 0             // terminal rows the drawn live partial occupies
    private var peak: Float = 0
    private var hasBuffers = true

    // Pinned bottom region, top→bottom: blank spacer · horizontal rule ·
    // brand+meter · device/stop. The spacer + rule visually separate the
    // scrolling transcript from the status footer.
    private static let footerHeight = 4

    init(tty: Bool) { self.tty = tty }

    private func out(_ s: String) { FileHandle.standardOutput.write(Data(s.utf8)) }

    func header(model: String, device: String, sampleRate: Int, channels: Int) {
        self.model = model
        self.device = device
        self.sr = sampleRate
        self.ch = channels
        rows = Term.rows
        guard tty, rows >= Self.footerHeight + 3 else {
            let suffix = model.isEmpty ? "" : " · \(model)"
            out("Voixful\(suffix) · listening — speak (Ctrl+C to stop)\n")
            return
        }
        enabled = true
        out("\u{1B}[2J\u{1B}[H")                              // clear screen, home
        out("\u{1B}[1;\(rows - Self.footerHeight)r")          // scroll region above footer
        // Park the cursor at the BOTTOM of the scroll region so live text
        // appears just above the footer and older lines scroll up — the
        // newest words stay where the eye is, next to the meter.
        out("\u{1B}[\(rows - Self.footerHeight);1H")
        liveRows = 0
        drawFooter()
    }

    func setPartial(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        render(t, seal: false)
    }

    func setLevel(_ p: Float, hasBuffers: Bool) {
        peak = p
        self.hasBuffers = hasBuffers
        if enabled { drawFooter() }
    }

    func commitFinal(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        if enabled {
            render(t, seal: true)
        } else {
            out(t + "\n")
        }
    }

    /// Reset the terminal: drop the scroll region and move below the footer so
    /// the shell prompt lands cleanly. The transcript stays on screen.
    func finish() {
        guard enabled else { return }
        out("\u{1B}[r")                       // reset scroll region (full screen)
        out("\u{1B}[\(rows);1H\n")            // below the footer
        enabled = false
    }

    /// Redraw the live partial IN PLACE. Partials are cumulative transcripts
    /// that get REVISED, not just extended (the full-context backends
    /// re-transcribe the whole utterance each refresh, and even Nemotron's
    /// detokenizer can retouch punctuation), so delta-append is wrong: any
    /// revision would spam a fresh line per refresh. Instead the current live
    /// block — which may span several wrapped rows — is erased (cursor up to
    /// its first row, clear to end of screen) and repainted. Long partials
    /// clamp to a tail window of a few rows ("…" prefix) so the block never
    /// outgrows what we can reliably erase inside the scroll region.
    /// `seal: true` paints the final text and newlines past it, so the next
    /// utterance starts a fresh block.
    private func render(_ text: String, seal: Bool) {
        guard enabled else { return }
        let cols = max(Term.cols, 10)
        if liveRows > 1 { out("\u{1B}[\(liveRows - 1)A") }  // to block's first row
        out("\r\u{1B}[0J")  // col 1, erase old block (+footer; redrawn below)
        if seal {
            out(text + "\n")     // final text flows + scrolls naturally
            liveRows = 0
        } else {
            let maxRows = min(6, max(1, rows - Self.footerHeight - 1))
            let maxChars = maxRows * cols - 1
            var shown = text
            if shown.count > maxChars {
                shown = "…" + String(shown.suffix(maxChars - 1))
            }
            out(shown)
            liveRows = max(1, (shown.count + cols - 1) / cols)
        }
        drawFooter()
    }

    private func drawFooter() {
        guard enabled else { return }
        let top = rows - Self.footerHeight + 1     // first row of the pinned region
        let rule = Term.dim + String(repeating: "─", count: max(10, Term.cols)) + Term.reset
        out("\u{1B}7")                                                  // save cursor
        out("\u{1B}[\(top);1H\u{1B}[2K")                                // blank spacer
        out("\u{1B}[\(top + 1);1H\u{1B}[2K\(rule)")                     // horizontal rule
        out("\u{1B}[\(rows - 1);1H\u{1B}[2K\(footerStatus())")          // brand + meter
        out("\u{1B}[\(rows);1H\u{1B}[2K\(footerInfo())")                // device + stop
        out("\u{1B}8")                                                  // restore cursor
    }

    private func footerStatus() -> String {
        let width = 18
        // Map peak to a dB scale so the meter spans the huge dynamic range of
        // real mics: quiet speech (peak ~0.007 ≈ -43 dB) still shows several
        // bars, silence (< -60 dB) shows none, loud speech fills it. A linear
        // gain can't do both — that's why the bar looked dead at 0.007.
        let db = 20 * log10f(max(peak, 1e-6))
        let level = min(max((db + 60) / 48, 0), 1)   // -60 dB → empty, -12 dB → full
        let filled = Int((level * Float(width)).rounded())
        let color = level > 0.85 ? 203 : (level > 0.6 ? 215 : 78)
        let meter = Term.fg(color) + String(repeating: "▮", count: filled)
            + Term.dim + String(repeating: "·", count: width - filled) + Term.reset
        let peakStr = String(format: "%.3f", peak)
        let warn = hasBuffers ? "" : "   \(Term.fg(203))no audio — check mic permission\(Term.reset)"
        // Brand, then the running model name (e.g. "Voixful · Nemotron").
        let title = model.isEmpty
            ? Term.brand()
            : "\(Term.brand()) \(Term.fg(240))·\(Term.reset) \(Term.fg(151))\(model)\(Term.reset)"
        return " \(title)  \(Term.fg(240))▕\(Term.reset)\(meter)\(Term.fg(240))▏\(Term.reset)"
            + " \(Term.dim)\(peakStr)\(Term.reset)\(warn)"
    }

    private func footerInfo() -> String {
        let dev = device.count > 34 ? String(device.prefix(33)) + "…" : device
        return " \(Term.dim)mic\(Term.reset) \(dev) · \(sr / 1000) kHz · \(ch)ch"
            + "   \(Term.fg(240))·\(Term.reset)   \(Term.dim)Ctrl+C to stop\(Term.reset)"
    }
}
