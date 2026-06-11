// nemotron-asr CLI — `file`, `mic`, and `languages` subcommands.
//
// Phase G implementation. Three subcommands:
//   * `file <wav>`     — transcribe a WAV file through `StreamingASR`, feeding
//                        100 ms slabs to exercise the streaming path (PLAN.md §7
//                        Phase 6 WER gate).
//   * `mic`            — capture from the default input device via
//                        AVAudioEngine, stream into `StreamingASR`, print
//                        partials and energy-VAD-driven utterance finals
//                        (PLAN.md §7 Phase 7 mic demo cadence).
//   * `languages`      — list the 40 supported locales (PLAN.md §2 / Types.swift
//                        `Language`).
//
// NOTE: macOS microphone access requires NSMicrophoneUsageDescription in the
// bundle's Info.plist when packaged. SwiftPM executables ask via the system
// prompt on first capture; if access is denied, AVAudioEngine.start throws.

import ArgumentParser
import AVFoundation
import Foundation
import NemotronASR

// MARK: - Root

@main
struct NemotronASRCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "nemotron-asr",
        abstract: "Streaming ASR via Core AI + Nemotron 3.5 — see PLAN.md §6.",
        subcommands: [FileCommand.self, MicCommand.self, LanguagesCommand.self, SmokeCommand.self]
    )
}

// MARK: - Shared option parsing helpers

/// Parse the `--chunk` integer flag (ms) into a `ChunkSize` case.
///
/// Throws an `ArgumentParser.ValidationError` listing the supported values if
/// the user supplied something other than {80, 160, 320, 560, 1120}.
private func parseChunkSize(_ ms: Int) throws -> ChunkSize {
    switch ms {
    case 80: return .ms80
    case 160: return .ms160
    case 320: return .ms320
    case 560: return .ms560
    case 1120: return .ms1120
    default:
        throw ValidationError(
            "Unsupported chunk size: \(ms) ms. Supported values: 80, 160, 320, 560, 1120."
        )
    }
}

/// Parse the `--language` raw value into a `Language` case.
///
/// Throws an `ArgumentParser.ValidationError` listing the supported values (one
/// per line, sorted) if the user supplied something the model doesn't carry.
private func parseLanguage(_ raw: String) throws -> Language {
    if let lang = Language(rawValue: raw) {
        return lang
    }
    let supported = Language.allCases.map { $0.rawValue }.sorted().joined(separator: ", ")
    throw ValidationError(
        "Unsupported language: '\(raw)'. Supported: \(supported)."
    )
}

// MARK: - stderr helpers

/// Write `msg` (with a trailing newline) to stderr. Used by error paths that
/// don't want to throw all the way up to ArgumentParser's default formatter
/// (e.g., AVFoundation conversion failures, mic-permission denials).
private func writeStderr(_ msg: String) {
    if let data = (msg + "\n").data(using: .utf8) {
        FileHandle.standardError.write(data)
    }
}

// MARK: - WAV loading + resampling

/// Load `path` via `AVAudioFile`, convert to 16 kHz mono Float32, and return
/// the samples as a flat `[Float]`.
///
/// If the file is already 16 kHz mono Float32, the AVAudioConverter path still
/// runs (it's a no-op fast path inside CoreAudio) — keeps the code uniform.
///
/// VERIFY: AVAudioConverter behavior on edge cases (24 kHz → 16 kHz with
/// non-integer ratio, 8 kHz upsample). The plan only commits to "16 kHz mono"
/// at `StreamingASR.feed`'s input. Phase 1 should add a WAV test fixture with
/// a non-16 kHz source rate to confirm we don't lose the tail.
private func loadWavAsFloat16kMono(path: String) throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let inputFile = try AVAudioFile(forReading: url)
    let inputFormat = inputFile.processingFormat

    // Target format: 16 kHz mono Float32 non-interleaved — the input shape
    // StreamingASR expects (PLAN.md §5 mel preprocessor is locked at 16 kHz).
    guard let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: false
    ) else {
        throw ASRError.unsupportedAudioFormat(
            "Could not construct target AVAudioFormat (16 kHz mono Float32)."
        )
    }

    // Fast path: already 16 kHz mono Float32. Read straight into a buffer and
    // copy out the samples.
    if inputFormat.sampleRate == 16_000
        && inputFormat.channelCount == 1
        && inputFormat.commonFormat == .pcmFormatFloat32
    {
        let frameCount = AVAudioFrameCount(inputFile.length)
        guard let buf = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCount) else {
            throw ASRError.unsupportedAudioFormat("Could not allocate AVAudioPCMBuffer.")
        }
        try inputFile.read(into: buf)
        return floatChannelData(buf)
    }

    // Slow path: convert via AVAudioConverter. Use a streaming converter loop
    // because the file may be larger than the converter's preferred internal
    // chunk size.
    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
        throw ASRError.unsupportedAudioFormat(
            "Could not construct AVAudioConverter \(inputFormat) → \(outputFormat)."
        )
    }

    // Choose a read chunk: 1 second of input audio. Large enough to amortize
    // the per-call overhead, small enough that we don't allocate excessive
    // peak memory for long files.
    let inputChunkFrames = AVAudioFrameCount(inputFormat.sampleRate)
    guard let inputBuffer = AVAudioPCMBuffer(
        pcmFormat: inputFormat,
        frameCapacity: inputChunkFrames
    ) else {
        throw ASRError.unsupportedAudioFormat("Could not allocate input AVAudioPCMBuffer.")
    }

    // Output buffer sized for the worst-case sample-rate ratio + slack.
    let outputCapacity = AVAudioFrameCount(
        Double(inputChunkFrames) * (16_000.0 / inputFormat.sampleRate) + 1024.0
    )
    guard let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: outputCapacity
    ) else {
        throw ASRError.unsupportedAudioFormat("Could not allocate output AVAudioPCMBuffer.")
    }

    var samples: [Float] = []
    samples.reserveCapacity(Int(Double(inputFile.length) * 16_000.0 / inputFormat.sampleRate))

    var endOfFile = false
    while !endOfFile {
        // The converter pulls from this closure. We read one chunk from the
        // file per pull; when the file is exhausted, we signal end-of-stream.
        var readError: Error?
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            do {
                try inputFile.read(into: inputBuffer)
                if inputBuffer.frameLength == 0 {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                outStatus.pointee = .haveData
                return inputBuffer
            } catch {
                readError = error
                outStatus.pointee = .endOfStream
                return nil
            }
        }

        var convError: NSError?
        let status = converter.convert(to: outputBuffer, error: &convError, withInputFrom: inputBlock)
        if let readError = readError { throw readError }
        if let convError = convError { throw convError }

        switch status {
        case .haveData:
            samples.append(contentsOf: floatChannelData(outputBuffer))
        case .inputRanDry:
            samples.append(contentsOf: floatChannelData(outputBuffer))
            // Will be re-pulled; next iteration's inputBlock may return endOfStream.
        case .endOfStream:
            samples.append(contentsOf: floatChannelData(outputBuffer))
            endOfFile = true
        case .error:
            throw ASRError.unsupportedAudioFormat("AVAudioConverter returned .error.")
        @unknown default:
            throw ASRError.unsupportedAudioFormat("AVAudioConverter returned unknown status.")
        }
    }

    return samples
}

/// Extract channel 0 of an `AVAudioPCMBuffer` as a `[Float]`. Assumes the
/// buffer is non-interleaved Float32 (which is what we construct above).
private func floatChannelData(_ buffer: AVAudioPCMBuffer) -> [Float] {
    let frameCount = Int(buffer.frameLength)
    guard frameCount > 0, let channels = buffer.floatChannelData else { return [] }
    let ptr = channels[0]
    return Array(UnsafeBufferPointer(start: ptr, count: frameCount))
}

// MARK: - `file` subcommand

struct FileCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "file",
        abstract: "Transcribe a WAV file via StreamingASR."
    )

    @Argument(help: "Path to input WAV file")
    var wav: String

    @Option(help: "Path to .aimodel directory")
    var model: String

    @Option(name: .shortAndLong, help: "Language code, e.g., en-US")
    var language: String = "en-US"

    @Option(name: .shortAndLong, help: "Chunk size in ms (80, 160, 320, 560, 1120)")
    var chunk: Int = 320

    @Flag(help: "Strip language tags from output")
    var stripLangTags: Bool = false

    func run() async throws {
        let chunkSize = try parseChunkSize(chunk)
        let lang = try parseLanguage(language)

        // Load + resample the WAV to 16 kHz mono Float32 before starting the
        // session — we want the load error to fire before we touch the model.
        let samples: [Float]
        do {
            samples = try loadWavAsFloat16kMono(path: wav)
        } catch {
            writeStderr("error: failed to load WAV '\(wav)': \(error)")
            throw ExitCode.failure
        }

        // Build the streaming session. Any model-load failure surfaces here.
        let asr: StreamingASR
        do {
            asr = try await StreamingASR(
                modelURL: URL(fileURLWithPath: model),
                language: lang,
                chunkSize: chunkSize
            )
        } catch {
            writeStderr("error: failed to load model '\(model)': \(error)")
            throw ExitCode.failure
        }

        await asr.start()

        // Feed in 100 ms slabs (1600 samples each @ 16 kHz) to exercise the
        // streaming path. Smaller than any chunkSize.totalFrames * hop so the
        // mel-frame buffering / encoder dispatch logic in `StreamingASR.feed`
        // runs the realistic streaming loop, not a single big batch.
        //
        // VERIFY: 1600-sample slab is intentionally hardcoded; it's a
        // streaming-path-exercise convention, not a model parameter. Phase 7's
        // mic path naturally produces slabs near this size (AVAudioEngine tap
        // typical buffer is ~1024-4096 frames @ HW rate, post-conversion).
        let slabSamples = 1600
        var idx = 0
        while idx < samples.count {
            let end = min(idx + slabSamples, samples.count)
            let slab = Array(samples[idx..<end])
            do {
                _ = try await asr.feed(slab)
            } catch {
                writeStderr("error: feed(...) failed at offset \(idx): \(error)")
                throw ExitCode.failure
            }
            idx = end
        }

        let transcript: Transcript
        do {
            transcript = try await asr.finish()
        } catch {
            writeStderr("error: finish() failed: \(error)")
            throw ExitCode.failure
        }

        // PLAN.md §7 Phase 6 WER gate measures `transcript.text` against the
        // NeMo reference; `--strip-lang-tags` is for human readers (the
        // tokenizer's `stripLangTags: true` already runs in `finish()`, but
        // belt-and-suspenders if the model emits anything the regex misses).
        //
        // VERIFY: whether `StreamingASR.finish()` already strips lang tags
        // unconditionally (StreamingASR.swift:417 says `stripLangTags: true`),
        // making this flag effectively a no-op. If the user wants raw tags,
        // we'd need a new `finish(stripLangTags:)` overload — leave the flag
        // for forward compatibility but document the current behavior.
        var text = transcript.text
        if stripLangTags {
            text = stripBracketedLangTags(text)
        }
        print(text)
    }
}

/// Belt-and-suspenders lang-tag stripper for the CLI's `--strip-lang-tags`
/// flag. The tokenizer already strips these in `decode(...)` when asked, but
/// this catches any that slipped through (e.g., if the tokenizer's stub mode
/// happened to detokenize a tag that didn't match its regex).
private func stripBracketedLangTags(_ s: String) -> String {
    let pattern = #"<[a-z]{2}-[A-Z]{2}>"#
    return s.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        .replacingOccurrences(of: "  ", with: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - Mic audio buffer (actor-isolated)

/// Thread-safe FIFO for audio samples flowing from the AVAudioEngine tap
/// (real-time thread) to the StreamingASR consumer Task.
///
/// The tap callback writes into this actor from its background thread; the
/// consumer Task awaits `drain()` to pick up whatever has accumulated. Actor
/// isolation gives us the synchronization we need without locks.
///
/// We keep the actor's surface tiny because every cross-actor call from the
/// tap callback hops to the actor's executor — we want as few hops per
/// callback as possible.
actor MicBuffer {
    private var samples: [Float] = []
    private var closed: Bool = false

    /// Append samples from the tap callback. Caller is responsible for having
    /// already done the heavy work (format conversion); this just stores.
    func append(_ newSamples: [Float]) {
        guard !closed else { return }
        samples.append(contentsOf: newSamples)
    }

    /// Pull all currently-buffered samples and clear the buffer. Returns an
    /// empty array if the buffer is empty.
    func drain() -> [Float] {
        let out = samples
        samples.removeAll(keepingCapacity: true)
        return out
    }

    /// Mark the buffer as closed. After this point `append` becomes a no-op
    /// and the consumer Task should drain once more and exit.
    func close() {
        closed = true
    }

    /// Whether the buffer has been closed AND drained.
    func isClosed() -> Bool { closed }
}

// MARK: - Mic Ctrl+C handling

/// Global mutable state for the signal handler. We can't capture the per-run
/// state in a sigaction callback (those are C function pointers with no
/// closure storage), so we stash a "please exit gracefully" flag and let the
/// consumer Task notice it.
nonisolated(unsafe) private var micShouldExit: Bool = false

/// Install a SIGINT (Ctrl+C) handler that sets `micShouldExit = true`. The
/// consumer Task polls this between drains and finalizes cleanly.
///
/// We use the plain `signal(2)` C API rather than `sigaction(2)` because:
///   1. The handler body only sets a global flag — no need for `siginfo_t`.
///   2. Swift requires C-function-pointer compatibility for the handler; the
///      `@convention(c)` closure below only references a top-level global, so
///      it has no implicit captures and is convertible to a C function pointer.
///
/// VERIFY: signal-handler safety. Setting `micShouldExit = true` is the only
/// side effect; Bool stores are atomic on every Darwin architecture we ship
/// to, so the consumer Task observing it via plain read is safe. If we ever
/// extend the handler to do more (e.g., logging), promote to `sigaction(2)`
/// with `sigaction.__sigaction_u.__sa_handler` per the Darwin SDK shape.
private func installMicSignalHandler() {
    let handler: @convention(c) (Int32) -> Void = { _ in
        micShouldExit = true
    }
    signal(SIGINT, handler)
}

// MARK: - `mic` subcommand

struct MicCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mic",
        abstract: "Stream from the default input device and transcribe live."
    )

    @Option(help: "Path to .aimodel directory")
    var model: String

    @Option(name: .shortAndLong, help: "Language code, e.g., en-US")
    var language: String = "en-US"

    @Option(name: .shortAndLong, help: "Chunk size in ms (80, 160, 320, 560, 1120)")
    var chunk: Int = 320

    func run() async throws {
        let chunkSize = try parseChunkSize(chunk)
        let lang = try parseLanguage(language)

        let asr: StreamingASR
        do {
            asr = try await StreamingASR(
                modelURL: URL(fileURLWithPath: model),
                language: lang,
                chunkSize: chunkSize
            )
        } catch {
            writeStderr("error: failed to load model '\(model)': \(error)")
            throw ExitCode.failure
        }

        await asr.start()

        let buffer = MicBuffer()

        // Target format: 16 kHz mono Float32 (what `StreamingASR.feed` wants).
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ) else {
            writeStderr("error: could not construct target AVAudioFormat for mic input.")
            throw ExitCode.failure
        }

        // AVAudioEngine setup. The input node's format is whatever the
        // hardware presents (typically 48 kHz stereo on Mac). We tap at the
        // input format and convert per-buffer inside the tap.
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)

        // Sanity: macOS sometimes returns a 0-channel format when no input
        // device is selected; surface a useful error rather than crashing in
        // installTap.
        guard inputFormat.channelCount > 0, inputFormat.sampleRate > 0 else {
            writeStderr(
                "error: input device reports an invalid format (channelCount=\(inputFormat.channelCount), "
                + "sampleRate=\(inputFormat.sampleRate)). Is a microphone configured?"
            )
            throw ExitCode.failure
        }

        // One converter for the lifetime of the engine; reused across every
        // tap callback. Building it once avoids the per-callback hot-path
        // construction cost.
        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            writeStderr(
                "error: could not construct AVAudioConverter \(inputFormat) → \(targetFormat)."
            )
            throw ExitCode.failure
        }

        // Install the tap. The closure runs on a real-time thread; we keep
        // its work as minimal as possible:
        //   1. Run the AVAudioConverter (CoreAudio-internal, fast).
        //   2. Copy the resulting Float32 samples out into a local [Float].
        //   3. Post to the actor with `Task { await buffer.append(...) }`.
        //
        // All heavy work (ASR feed, VAD, partial-emission) lives in the
        // consumer Task below. The tap MUST NOT call `await` directly; it's
        // a synchronous callback.
        //
        // VERIFY: AVAudioEngine tap buffer size — 4096 frames is a common safe
        // default that minimizes per-callback overhead while keeping latency
        // reasonable (~85 ms @ 48 kHz hardware rate, ~256 ms @ 16 kHz). If RTF
        // measurements in PLAN.md §7 Phase 7 require a tighter cadence, drop
        // to 1024 frames (~21 ms @ 48 kHz). AVAudioEngine may clamp the
        // requested size to a hardware-permitted value; the actual delivered
        // buffer size is `inputBuffer.frameLength`, not this argument.
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { inputBuffer, _ in
            // Convert in-place into a freshly-allocated output buffer sized
            // for the worst-case ratio + slack.
            let outputCapacity = AVAudioFrameCount(
                Double(inputBuffer.frameLength) * (16_000.0 / inputFormat.sampleRate) + 256.0
            )
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: outputCapacity
            ) else {
                // Allocation failed on the audio thread — drop the buffer
                // silently rather than tearing down the stream.
                return
            }

            var convError: NSError?
            var inputConsumed = false
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                if inputConsumed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                inputConsumed = true
                outStatus.pointee = .haveData
                return inputBuffer
            }
            let status = converter.convert(
                to: outputBuffer, error: &convError, withInputFrom: inputBlock
            )
            if status == .error || convError != nil {
                return
            }
            let samples = floatChannelData(outputBuffer)
            if samples.isEmpty { return }

            // Hand off to the actor. Spawning a `Task` from the audio thread
            // is the documented pattern for posting to an actor from a
            // non-async context; the Task body runs on the actor's executor.
            Task { await buffer.append(samples) }
        }

        // Install Ctrl+C handler. Sets a global flag the consumer Task polls.
        installMicSignalHandler()

        do {
            try engine.start()
        } catch {
            writeStderr(
                "error: AVAudioEngine.start() failed: \(error). "
                + "If the system did not prompt for microphone access, "
                + "ensure NSMicrophoneUsageDescription is set in Info.plist when packaging."
            )
            throw ExitCode.failure
        }

        // Consumer loop. Pulls from the buffer, feeds StreamingASR, prints
        // partials, and runs energy-based VAD to detect utterance boundaries.
        //
        // VAD: 20 ms windows (320 samples @ 16 kHz). Track silence run length;
        // if > 500 ms of silence after at least some speech, treat as
        // end-of-utterance — call finish, print transcript, start a fresh
        // session.
        //
        // VAD constants live in `VADDefaults` so the SwiftUI demo and the CLI
        // mic subcommand share one source of truth. The 0.005 RMS / 500 ms
        // silence values are placeholders; see PLAN.md §7 Phase 7 / §10 open
        // question on VAD strategy.
        let vadWindowSamples = VADDefaults.windowSamples
        let energyThreshold = VADDefaults.energyThreshold
        let silenceWindowsToFinish = VADDefaults.silenceWindowsToFinish

        var silenceWindowRun = 0
        var hasSeenSpeech = false
        var leftover: [Float] = []  // for sub-vad-window remainder

        // Loop until Ctrl+C, then drain once more and finalize.
        while !micShouldExit {
            let pulled = await buffer.drain()
            if pulled.isEmpty {
                // No new audio yet — yield to avoid spinning. Sleep ~20 ms
                // (one VAD window's worth); the AVAudioEngine tap typically
                // posts every 20-40 ms anyway.
                try? await Task.sleep(nanoseconds: 20_000_000)
                continue
            }

            // Feed to ASR. This runs the mel extractor + encoder + decoder.
            do {
                let partials = try await asr.feed(pulled)
                for partial in partials {
                    print("[partial] \(partial.text)")
                }
            } catch {
                writeStderr("error: feed(...) failed: \(error)")
                // Tear down rather than spin on a recurring error.
                engine.stop()
                inputNode.removeTap(onBus: 0)
                throw ExitCode.failure
            }

            // VAD: compute RMS per 20 ms window over (leftover + pulled).
            var combined = leftover
            combined.append(contentsOf: pulled)
            var w = 0
            while w + vadWindowSamples <= combined.count {
                let rms = computeRMS(samples: combined, start: w, length: vadWindowSamples)
                if rms >= energyThreshold {
                    hasSeenSpeech = true
                    silenceWindowRun = 0
                } else if hasSeenSpeech {
                    silenceWindowRun += 1
                    if silenceWindowRun >= silenceWindowsToFinish {
                        // End-of-utterance: finalize, print, start fresh.
                        do {
                            let transcript = try await asr.finish()
                            print(transcript.text)
                        } catch {
                            writeStderr("error: finish() failed at utterance boundary: \(error)")
                        }
                        await asr.start()
                        silenceWindowRun = 0
                        hasSeenSpeech = false
                    }
                }
                w += vadWindowSamples
            }
            leftover = Array(combined[w..<combined.count])
        }

        // Ctrl+C path: drain remaining audio, finalize once, print, exit 0.
        engine.stop()
        inputNode.removeTap(onBus: 0)
        await buffer.close()

        let tail = await buffer.drain()
        if !tail.isEmpty {
            _ = try? await asr.feed(tail)
        }

        do {
            let transcript = try await asr.finish()
            print(transcript.text)
        } catch {
            writeStderr("error: final finish() failed: \(error)")
            throw ExitCode.failure
        }
    }
}

/// RMS of `samples[start..<start+length]`. Stays in plain Swift because the
/// mic VAD's hot path is 50 calls/sec at 320 samples each — cheap enough that
/// a vDSP_rmsqv hop isn't worth the additional import surface on a CLI.
///
/// If this becomes a bottleneck (it won't), replace with `vDSP_rmsqv`.
private func computeRMS(samples: [Float], start: Int, length: Int) -> Float {
    var sumSquares: Float = 0
    let end = start + length
    var i = start
    while i < end {
        let s = samples[i]
        sumSquares += s * s
        i += 1
    }
    return (sumSquares / Float(length)).squareRoot()
}

// MARK: - `languages` subcommand

struct LanguagesCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "languages",
        abstract: "List the 40 supported locale tags."
    )

    func run() async throws {
        let tags = Language.allCases.map { $0.rawValue }.sorted().joined(separator: "\n")
        print(tags)
    }
}

// MARK: - `smoke` subcommand

/// End-to-end wiring smoke test. The four model-free steps (enum sanity, mel
/// extractor on synthetic audio, language one-hot, and an optional full
/// pipeline pass if `--model` is supplied) exercise everything Phase A-G wired
/// up without needing real model artifacts. Designed to complete in under 10
/// seconds when the environment is ready.
///
/// Step implementations live in `NemotronASR.Smoke` (Sources/NemotronASR/Smoke.swift)
/// so `SmokeTests` and this subcommand exercise the same code.
struct SmokeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "smoke",
        abstract: "Run end-to-end wiring checks. Uses synthetic audio; loads model only if --model is provided."
    )

    @Option(help: "Optional path to .aimodel — exercises the full pipeline if provided.") var model: String?
    @Option(name: .shortAndLong, help: "Language code") var language: String = "en-US"
    @Option(name: .shortAndLong, help: "Chunk size in ms") var chunk: Int = 320

    func run() async throws {
        print("[smoke] Nemotron ASR + Core AI — Swift smoke driver")

        var stepsPassed = 0
        var stepsFailed = 0
        var stepsSkipped = 0

        func record(_ stepLabel: String, _ result: SmokeStepResult) {
            print(smokeReport(step: stepLabel, result: result))
            switch result.status {
            case .pass: stepsPassed += 1
            case .fail: stepsFailed += 1
            case .skipped: stepsSkipped += 1
            }
        }

        // Step 1: Enum sanity (no model, no audio).
        record("1", Smoke.step1EnumSanity())

        // Step 2: Synthetic audio through MelExtractor.
        let step2 = await Smoke.step2MelExtractor()
        record("2", step2)

        // Step 3: LangID.oneHot.
        let parsedLanguage: Language
        do {
            parsedLanguage = try parseLanguage(language)
        } catch {
            record("3", .fail("parseLanguage(\(language)) threw: \(error)"))
            // We can still proceed to step 4 with the default; but step 4 also
            // needs the parsed language, so jump to the summary.
            print("[smoke] summary: \(stepsPassed) passed, \(stepsFailed) failed, \(stepsSkipped) skipped")
            throw ExitCode.failure
        }
        record("3", Smoke.step3LangIDOneHot(parsedLanguage))

        // Step 4: optional full pipeline. Only runs if --model was passed and
        // the path exists on disk; otherwise we record SKIPPED.
        if let modelPath = model {
            let url = URL(fileURLWithPath: modelPath)
            if !FileManager.default.fileExists(atPath: url.path) {
                record("4", .skipped("--model path does not exist: \(modelPath)"))
            } else {
                record("4", await runStep4FullPipeline(
                    modelPath: modelPath,
                    language: parsedLanguage,
                    chunkMS: chunk
                ))
            }
        } else {
            record("4", .skipped("no --model provided; skipping full-pipeline check"))
        }

        // Step 5: summary.
        let summary = "[smoke] summary: \(stepsPassed) passed, \(stepsFailed) failed, \(stepsSkipped) skipped"
        print(summary)
        if stepsFailed > 0 {
            throw ExitCode.failure
        }
    }
}

/// Step 4: optional full-pipeline smoke. Lives in the CLI module because
/// `StreamingASR` is constructed here and we want the CLI's
/// `parseChunkSize`/error-formatting conventions on this path. The test target
/// exercises Steps 1-3 only; Step 4 needs real model artifacts and is the one
/// step we want to skip in CI without a `.aimodel` fixture.
private func runStep4FullPipeline(
    modelPath: String,
    language: Language,
    chunkMS: Int
) async -> SmokeStepResult {
    let chunkSize: ChunkSize
    do {
        chunkSize = try parseChunkSize(chunkMS)
    } catch {
        return .fail("parseChunkSize(\(chunkMS)) threw: \(error)")
    }

    let asr: StreamingASR
    do {
        asr = try await StreamingASR(
            modelURL: URL(fileURLWithPath: modelPath),
            language: language,
            chunkSize: chunkSize
        )
    } catch {
        return .fail("StreamingASR init threw: \(error)")
    }

    await asr.start()

    // Feed 2 seconds of synthetic audio in 100 ms slabs.
    let oneSecond = Smoke.synthesizeOneSecondAudio()
    let twoSeconds = oneSecond + oneSecond
    let slabSize = 1600  // 100 ms @ 16 kHz
    var idx = 0
    while idx < twoSeconds.count {
        let end = min(idx + slabSize, twoSeconds.count)
        let slab = Array(twoSeconds[idx..<end])
        do {
            _ = try await asr.feed(slab)
        } catch {
            return .fail("StreamingASR.feed at offset \(idx) threw: \(error)")
        }
        idx = end
    }

    // We don't assert on the text content — a 440 Hz sine in silence has no
    // defined transcript and the model may emit empty or garbage. Just confirm
    // we got a `Transcript` back without throwing.
    do {
        let transcript = try await asr.finish()
        let charCount = transcript.text.count
        return .pass(
            "StreamingASR end-to-end on 2 s synthetic audio — transcript \(charCount) chars, "
            + "language=\(transcript.language ?? "nil")"
        )
    } catch {
        return .fail("StreamingASR.finish threw: \(error)")
    }
}
