// Phase F — Public StreamingASR actor.
//
// The package's single public entry point for streaming transcription. Owns
// the four sessions (encoder, prediction, joiner, mel extractor + tokenizer)
// and the small per-utterance buffers that glue them together.
//
// Public API — exactly four methods, matching PLAN.md §6 "Public API sketch":
//
//   public init(modelURL: URL, language: Language = .enUS, chunkSize: ChunkSize = .ms320) async throws
//   public func start() async
//   public func feed(_ samples: [Float]) async throws -> [PartialHypothesis]
//   public func finish() async throws -> Transcript
//
// No other methods are public. Internal state (mel/audio buffers, emitted
// token accumulator, the language one-hot NDArray) is accessible only via
// these four entry points so callers can't accidentally desync the cache
// state across the four sessions (Anti-pattern 11 in PLAN.md §11).
//
// Lifecycle:
//   1. `init(...)` loads the `.aimodel` once, builds the four sessions and
//      the mel extractor + tokenizer. Constructs the language one-hot NDArray
//      once per session (immutable for the session's lifetime).
//   2. `start()` resets all four sessions and the mel extractor to a fresh
//      utterance state. Idempotent: calling it twice in a row has the same
//      effect as calling it once.
//   3. `feed(samples:)` appends raw 16 kHz mono samples to the audio buffer,
//      runs the mel extractor in streaming mode, and — once enough mel frames
//      are buffered to fill a chunk — calls the encoder, decoder, and tokenizer
//      to produce zero or more `PartialHypothesis` for this call.
//   4. `finish()` flushes any remaining audio / mel frames through one final
//      encoder pass with right-context padding, then returns the full
//      Transcript.
//
// chunkSize is stored at init; the encoder does NOT take it as an input
// (`att_context_size` is baked into the model — but if convert.py exports a
// dynamic-shape model per PLAN.md §10's open question, chunkSize only affects
// how many mel frames we accumulate before invoking the encoder).

import CoreAI
import Foundation

public actor StreamingASR {
    // MARK: - Configuration

    /// Locale this session was constructed with. Drives the one-hot vector
    /// fed to the encoder on every call; immutable for the session's lifetime
    /// (re-instantiate for a different language).
    public let language: Language

    /// Chunk size this session was constructed with. Drives how many mel
    /// frames we accumulate before invoking the encoder; immutable for the
    /// session's lifetime.
    public let chunkSize: ChunkSize

    // MARK: - Sessions

    private let preprocess: PreprocessSession
    private let encoder: EncoderSession
    private let prediction: PredictionSession
    private let joiner: JoinerSession
    private let decoder: GreedyRNNTDecoder
    private let melExtractor: MelExtractor
    private let tokenizer: Tokenizer

    // MARK: - Per-utterance state

    /// Raw audio samples awaiting mel extraction. Appended on each `feed(...)`
    /// call; drained by the streaming mel extractor (which keeps its own
    /// hop-overlap carryover, so we hand off complete frames to it and let it
    /// manage the inter-call overlap).
    ///
    /// VERIFY: whether passing audio through this intermediate buffer and ALSO
    /// letting `MelExtractor.extractStreaming` keep its own carryover is
    /// double-buffering. PLAN.md §5 / Phase C review notes that the mel
    /// extractor's streaming carryover holds the windowSize-hopSize overlap
    /// (~240 samples). Phase 1 may simplify this to "audioBuffer holds
    /// nothing; pass directly to extractStreaming" once the mel extractor's
    /// streaming contract is locked in.
    private var audioBuffer: [Float] = []

    /// Mel frames accumulated since the last encoder call, in row-major
    /// `[nMels × T]` order matching `MelExtractor.extractStreaming(...)`'s
    /// return value. Drained when we have at least `chunkSize.totalFrames`
    /// frames available and the encoder runs.
    ///
    /// Stored as a flat `[Float]` rather than a 2D structure because that's
    /// what `MelExtractor` returns and what we need to repack into an
    /// `NDArray` of shape `[1, 128, T]` for the encoder. Keeping it flat
    /// avoids an extra copy/transpose on every feed.
    private var melBuffer: [Float] = []

    /// Number of frames currently sitting in `melBuffer`. Tracked separately
    /// because `melBuffer.count == nMels * melBufferFrames`.
    private var melBufferFrames: Int = 0

    /// All non-blank token ids emitted across this utterance, in emit order.
    /// Used to build the final `Transcript` text in `finish()`. The decoder
    /// also maintains its own accumulator (`GreedyRNNTDecoder.emittedTokens`);
    /// we duplicate here because the decoder's accumulator is private and
    /// `finish()` needs to detokenize the whole utterance.
    private var allEmittedTokens: [Int32] = []

    /// Wall-clock timestamp (CFAbsoluteTime seconds) when `start()` was last
    /// called. Used to compute `PartialHypothesis.timestamp`.
    private var sessionStartTime: TimeInterval = 0

    // MARK: - Immutable per-session inputs

    /// 128-dim language one-hot, built once at init via `LangID.oneHot(language)`.
    /// Reused on every encoder call — never reallocated, never mutated (the
    /// encoder reads it as a regular input, not a state buffer, so the
    /// MutableViews zero-copy semantic doesn't apply, but reusing the same
    /// NDArray instance still avoids per-chunk allocation per Anti-pattern 4
    /// in PLAN.md §11).
    private let langOneHot: NDArray

    // MARK: - Mel preprocessor constants

    /// Mel bins. Mirrored from `MelExtractor`'s default (128 per PLAN.md §5).
    /// Stored locally so we don't `await` into the mel extractor actor every
    /// time we need to repack frames into the encoder's `[1, 128, T]` shape.
    private static let nMels: Int = 128

    /// The fixed mel-frame count the converted `.aimodel` accepts. The current
    /// convert.py pins T=128 (smallest multiple of 8 ≥ 121, the streaming-cfg
    /// minimum) because coreai_torch 0.4.0 hits a dynamic-shape lowering bug
    /// on `aten.repeat`. Once that's fixed upstream we can re-introduce the
    /// `ChunkSize` dial and export a variable-T encoder.
    ///
    /// `ChunkSize` is still surfaced on the public API for forward compat —
    /// today every chunk-size case routes through this same T=128 hardcode.
    private static let encoderFramesPerChunk: Int = 128

    // MARK: - Init

    /// Construct a `StreamingASR` session.
    ///
    /// - Parameters:
    ///   - modelURL: Path to the `nemotron-streaming.aimodel` artifact
    ///     produced by `convert/nemotron_convert/convert.py`. The vocabulary
    ///     file is expected as a sibling named `vocab.txt` (see VERIFY below).
    ///   - language: Target locale for transcription. Drives the prompt-kernel
    ///     language conditioning via a 128-dim one-hot vector (PLAN.md §2).
    ///     Defaults to `.enUS`.
    ///   - chunkSize: Cache-aware streaming chunk size (PLAN.md §2's
    ///     "Chunk-size dial"). Defaults to `.ms320` per PLAN.md §9 Decision 9.
    /// - Throws: `ASRError.modelLoadFailed` if the `.aimodel` cannot be
    ///   loaded; `ASRError.missingFunction` if any of the three required
    ///   inference functions ("encode", "predict_step", "join") are missing.
    public init(
        modelURL: URL,
        language: Language = .enUS,
        chunkSize: ChunkSize = .ms320
    ) async throws {
        self.language = language
        self.chunkSize = chunkSize

        // Load the multi-function .aimodel exactly once. The three sessions
        // share this model instance; each one loads its named function.
        //
        // VERIFY: Core AI API surface — `AIModel(contentsOf:)` is the
        // initializer shown in PLAN.md §6 "Model loading". Whether it's
        // `async throws` or `throws`-only is unconfirmed against Xcode 27 SDK;
        // PLAN.md §6 uses `try await AIModel(contentsOf:)` so we mirror that.
        // If the SDK turns out to be sync, remove the `await` here AND in
        // any error path that bridges to ASRError.modelLoadFailed.
        // Use AIModelCache with .persistent policy so the device-specific
        // specialization (the cold-start cost — measured at ~970ms on first
        // call, ~200ms warm) is reused across process launches. First launch
        // after install: ~1s. Every subsequent launch: ~200ms.
        let model: AIModel
        do {
            if let cached = try AIModelCache.default.model(for: modelURL, options: .default) {
                model = cached
            } else {
                model = try await AIModel.specialize(
                    contentsOf: modelURL,
                    options: .default,
                    cache: AIModelCache.default,
                    cachePolicy: .persistent
                )
            }
        } catch {
            throw ASRError.modelLoadFailed(
                "AIModel.specialize(contentsOf: \(modelURL.path)) failed: \(error)"
            )
        }

        // Build the three Core AI sessions in dependency order. EncoderSession
        // and PredictionSession each own state buffers; JoinerSession is
        // stateless. Any of these can throw `missingFunction` if the .aimodel
        // doesn't contain the named function — that error propagates here as-is.
        self.preprocess = try await PreprocessSession(model: model)
        self.encoder = try await EncoderSession(model: model)
        self.prediction = try await PredictionSession(model: model)
        self.joiner = try await JoinerSession(model: model)

        // Build the greedy RNN-T decoder. Holds references to prediction and
        // joiner sessions; does NOT hold the encoder (encoder outputs are
        // passed into `decode(encoderOutput:)`) — see GreedyRNNTDecoder.swift
        // header.
        self.decoder = GreedyRNNTDecoder(
            prediction: self.prediction,
            joiner: self.joiner
        )

        // Build the mel extractor with PLAN.md §5 defaults (16 kHz, 128 mels,
        // 400-sample window, 160-sample hop, 512-point FFT, Slaney mel scale).
        self.melExtractor = MelExtractor()

        // Build the tokenizer. The vocab artifact is expected as a sibling of
        // the .aimodel file.
        //
        // VERIFY: vocab artifact convention. PLAN.md doesn't lock in the
        // distribution shape for the SentencePiece vocab — it could be:
        //   1. A sibling `vocab.txt` (what we assume here)
        //   2. A sibling `tokenizer.model` (SentencePiece binary)
        //   3. Bundled inside the .aimodel as an asset (Core AI assets
        //      support side files per the WWDC 2026 "Integrate on-device AI
        //      models" session)
        // Phase 1 / Phase 6 pick the convention. If it changes, update both
        // the `convert/` packaging step AND this URL construction.
        // Prefer pieces.txt (SentencePiece pieces, ▁-prefixed) if present.
        // vocab.txt in the .nemo archive is BERT-WordPiece-style (##-prefixed)
        // and does NOT match the SP token IDs the model emits — using it
        // produces garbage like "##b".
        let parentDir = modelURL.deletingLastPathComponent()
        let piecesURL = parentDir.appendingPathComponent("pieces.txt")
        let vocabURL = FileManager.default.fileExists(atPath: piecesURL.path)
            ? piecesURL
            : parentDir.appendingPathComponent("vocab.txt")
        self.tokenizer = try Tokenizer(vocabURL: vocabURL)

        // Build the language one-hot vector exactly once. Reused on every
        // encoder call for the session's lifetime — see Anti-pattern 4.
        self.langOneHot = LangID.oneHot(language)
    }

    // MARK: - start

    /// Reset all session state to a fresh utterance.
    ///
    /// Calls `reset()` on the encoder, prediction, decoder, and mel extractor
    /// in one transaction — per PLAN.md §4 / Anti-pattern 11, all caches reset
    /// together, never partially. Also clears the per-utterance audio buffer,
    /// mel buffer, and emitted-token accumulator, and records the session
    /// start timestamp for partial-hypothesis time-stamping.
    ///
    /// Idempotent: calling twice in a row has the same effect as calling once.
    public func start() async {
        // Reset all four stateful components together. Order doesn't matter
        // (they don't observe each other's state during reset), but doing
        // them in this fixed order makes the call site easy to audit.
        await encoder.reset()
        await prediction.reset()
        await decoder.reset()
        await melExtractor.reset()

        // Joiner is stateless; no reset() to call (intentionally — see
        // JoinerSession.swift).

        // SOS warmup. Our exported predict_step was built with add_sos=False
        // (NeMo's predict signature). Without an explicit SOS step the
        // prediction LSTM starts at zeros and the first joiner call sees
        // pred_emb=zeros, biasing toward blank — which loses the first
        // ~0.5s of speech in every utterance. NeMo's reference inference
        // achieves SOS-conditioning via predict(add_sos=True); we simulate
        // it by feeding the blank id once. In blank-as-SOS RNN-T
        // predictors (which this checkpoint is), processing blank from a
        // fresh state produces exactly the SOS embedding.
        do {
            try await prediction.step(token: GreedyRNNTDecoder.blankId)
        } catch {
            // If the SOS warmup fails, transcription degrades to "first
            // word(s) lost" — annoying but not crashing.
        }

        // Drop any partially-consumed audio / mel / token state from the
        // prior utterance.
        audioBuffer.removeAll(keepingCapacity: true)
        melBuffer.removeAll(keepingCapacity: true)
        melBufferFrames = 0
        allEmittedTokens.removeAll(keepingCapacity: true)

        // Stamp the session start. Used as the reference t=0 for
        // PartialHypothesis.timestamp.
        sessionStartTime = CFAbsoluteTimeGetCurrent()
    }

    // MARK: - feed

    /// Feed raw 16 kHz mono `Float` samples to the session. Returns any
    /// partial hypotheses produced while consuming the provided samples.
    ///
    /// Buffering semantic:
    ///   - The provided samples are appended to an internal audio buffer.
    ///   - The mel extractor runs in streaming mode on the new audio and
    ///     appends frames to an internal mel buffer.
    ///   - While the mel buffer holds at least `chunkSize.totalFrames` frames,
    ///     we slice off one chunk's worth of frames, repack into an
    ///     `NDArray` of shape `[1, 128, T]`, and drive
    ///     `encoder.forward → decoder.decode → tokenizer.decode` to produce
    ///     one `PartialHypothesis`.
    ///
    /// The audio MUST be 16 kHz mono Float — this is the rate the mel
    /// preprocessor was tuned for in PLAN.md §5. Sample-rate conversion,
    /// channel-down-mixing, and integer→float conversion happen in the
    /// caller (e.g., the CLI's AVAudioEngine plumbing in Phase 7).
    ///
    /// - Parameter samples: Raw audio samples in `[-1.0, 1.0]` range.
    /// - Returns: Zero or more `PartialHypothesis` produced by this call.
    ///   Empty if the call did not push enough audio to complete a chunk.
    /// - Throws: `ASRError.missingOutput` or session-specific errors if any
    ///   of the inference functions fail.
    public func feed(_ samples: [Float]) async throws -> [PartialHypothesis] {
        // Append the new audio to our buffer. The mel extractor manages its
        // own hop-overlap carryover, so we'll drain `audioBuffer` to it as
        // a single contiguous block per feed call.
        audioBuffer.append(contentsOf: samples)

        // Run the streaming mel extractor on whatever we have. Returns nil if
        // not enough samples for a full frame. Note: the mel extractor's
        // streaming carryover is INTERNAL — passing samples here once means
        // we should NOT re-pass them next call. So we drain `audioBuffer` to
        // the mel extractor and clear it.
        let newAudio = audioBuffer
        audioBuffer.removeAll(keepingCapacity: true)

        if let newMelFrames = await melExtractor.extractStreaming(samples: newAudio) {
            // newMelFrames is row-major [nMels × T_new] flattened. Number of
            // frames = total elements / nMels.
            let newFrames = newMelFrames.count / Self.nMels
            melBuffer.append(contentsOf: newMelFrames)
            melBufferFrames += newFrames
        }

        // Now consume as many chunks as we have. Each chunk needs
        // `chunkSize.totalFrames` mel frames.
        let framesPerChunk = Self.encoderFramesPerChunk
        var partials: [PartialHypothesis] = []

        while melBufferFrames >= framesPerChunk {
            // Slice off the first `framesPerChunk` frames from melBuffer.
            //
            // Mel storage is row-major `[nMels × melBufferFrames]`. The
            // encoder wants `[1, 128, T]`. The repack is:
            //   chunkMel[1, m, t] = melBuffer[m * melBufferFrames + t]   for t in 0..<framesPerChunk
            //
            // We can NOT just take the first `framesPerChunk * nMels`
            // elements: that would be row-major with the wrong row stride
            // (melBufferFrames, not framesPerChunk).
            let chunkMel = sliceMelChunk(frames: framesPerChunk)

            // Drive one encoder call. The encoder updates its four cache
            // tensors in place via MutableViews.
            // Pass mel_length matching the chunk's frame count. The English-only
            // checkpoint takes (mel_chunk, mel_length); the multilingual variant
            // would also take a `lang_id` here.
            var melLength = NDArray(shape: [1], scalarType: .int32)
            melLength.setInt32(at: 0, to: Int32(framesPerChunk))
            let encOut = try await encoder.forward(mel: chunkMel, melLength: melLength)

            // Drive the greedy decoder over the encoder output's time axis.
            // Returns just the tokens emitted by THIS decode call.
            let newTokens = try await decoder.decode(encoderOutput: encOut)

            // Track the utterance-wide accumulator for the final transcript.
            allEmittedTokens.append(contentsOf: newTokens)

            // Detokenize the utterance-so-far for the partial. We use the
            // full accumulator (not just newTokens) so the partial's `.text`
            // reads as a growing utterance, matching the "live captioning"
            // UX expectation of the partial-hypothesis API.
            //
            // VERIFY: PartialHypothesis API contract — PLAN.md §6 doesn't
            // pin down whether `text` is incremental ("new tokens only") or
            // cumulative ("utterance-so-far"). The current Phase D
            // PartialHypothesis doc-comment at Types.swift:43-49 says
            // "current best-effort transcription text", which reads as
            // cumulative. If the consumer expects incremental, change this
            // to `tokenizer.decode(tokens: newTokens, ...)` and update
            // Types.swift in lockstep.
            let text = await tokenizer.decode(tokens: allEmittedTokens, stripLangTags: true)

            // Stable-until-token: the right context for THIS chunk is
            // `chunkSize.attContextRight` frames; the tokens emitted at the
            // tail of `newTokens` may still be revised by the next chunk's
            // encoder pass once it sees that right context as "current".
            //
            // VERIFY: the stable/unstable split needs the chunk-aware
            // FastConformer's right-context semantics. PLAN.md §2 says the
            // right-context window is included in the chunk; whether THIS
            // call's emitted tokens at the right-context boundary can be
            // revised by the next call is an open question. Phase 7's
            // microphone demo (PLAN.md §7) will exercise this; a
            // conservative placeholder is to mark all tokens up to
            // `allEmittedTokens.count - newTokens.count` as stable.
            let stableUntil = allEmittedTokens.count - newTokens.count

            let now = CFAbsoluteTimeGetCurrent()
            partials.append(PartialHypothesis(
                text: text,
                stableUntilToken: stableUntil,
                timestamp: now - sessionStartTime
            ))
        }

        return partials
    }

    // MARK: - finish

    /// Close the utterance and return the final transcript.
    ///
    /// Flushes any remaining audio / mel frames through one final encoder
    /// pass — even if fewer than `chunkSize.totalFrames` mel frames are
    /// buffered, we right-context-pad the chunk with zeros and run the
    /// encoder so the decoder gets a chance to emit any pending tokens.
    ///
    /// Returns a `Transcript` whose `.text` is the detokenized full
    /// utterance, with one segment per non-blank token group (Phase F leaves
    /// segment time-alignment as a placeholder — single segment with the
    /// full text and total duration).
    ///
    /// Does NOT call `start()` internally; the caller drives the next
    /// utterance by calling `start()` again. This keeps the lifecycle
    /// explicit per the actor's documented contract.
    /// One-shot transcription for a complete recording. Bypasses the
    /// streaming mel path (no preemph-across-call carryover, no per-call
    /// reflect-pad gaps) by running the non-streaming `extract(samples:)` once
    /// on the whole audio buffer, then driving the encoder/decoder in
    /// 128-frame chunks. Use this for `nemotron-asr file` where the audio
    /// length is known up front; for live mic use `feed(...)` + `finish()`.
    /// Per-stage wall-clock counters (ms). Populated by `transcribe(audio:)`
    /// every call. Reset on `start()`. Read by callers to print breakdowns.
    public struct PerfCounters: Sendable {
        public var preprocessMs: Double = 0
        public var encodeMs: Double = 0
        public var decodeMs: Double = 0
        public var copyMs: Double = 0
        public var chunkCount: Int = 0
        public var audioMs: Double = 0
    }
    public private(set) var lastPerf = PerfCounters()

    public func transcribe(audio: [Float]) async throws -> Transcript {
        await self.start()
        var perf = PerfCounters()
        perf.audioMs = Double(audio.count) / 16.0  // ms (16 kHz)

        let chunkSamples = PreprocessSession.audioSamples
        let framesPerChunk = Self.encoderFramesPerChunk

        // Prepend silence chunks so the encoder's cache-aware K/V state
        // warms up before real speech hits. The cache window is 70 frames
        // (~0.7 s); each warmup chunk contributes ~14 frames. Without
        // warmup, the encoder produces "cold-cache" embeddings for the
        // first ~3 output frames (~0.3 s), the joiner emits blank for
        // those, and the first word(s) of the utterance get lost — even
        // with SOS predictor warmup. The extra chunks cost ~50 ms of
        // wall-clock each; 2 chunks is a good balance (recovers "The
        // Quick" on the bundled sample without bloating cold-start).
        let warmupChunks = 1
        var workingAudio = [Float](repeating: 0, count: warmupChunks * chunkSamples)
        workingAudio.append(contentsOf: audio)
        let audio = workingAudio  // shadow the parameter

        // Pipeline: while the decoder is chewing chunk N's encoder output on
        // the CPU + a stream of small ANE calls, we kick off chunk N+1's
        // preprocess + encoder on the ANE. The encoder is cache-aware so its
        // calls MUST be serial (each one reads the previous one's cache
        // state); we exploit that by running encoder N+1 immediately after
        // encoder N's await returns (state is committed), in a fresh Task.
        //
        // Wins: the decode's ~12 ms/chunk overlaps with the next chunk's
        // ~32 ms/chunk preprocess+encode. The savings depend on whether the
        // ANE serializes encoder+joiner calls at the hardware level (it
        // likely does in practice), so we measure rather than promise.

        // Pre-allocate input buffers once. NDArrays carry their own backing
        // store so we reuse them across chunks via mutableView() writes.
        var audioND = NDArray(shape: [1, chunkSamples], scalarType: .float32)
        var lengthND = NDArray(shape: [1], scalarType: .int32)
        lengthND.setInt32(at: 0, to: Int32(chunkSamples))
        var melLength = NDArray(shape: [1], scalarType: .int32)
        melLength.setInt32(at: 0, to: Int32(framesPerChunk))

        // Helper to do copy + preprocess + encode and return the encoder output.
        // Captures `audioND` etc. by-value via implicit copy.
        func runUpToEncoder(chunkStart: Int, realSamples: Int) async throws -> (NDArray, Double, Double, Double) {
            // 1. Sample copy (CPU).
            let cpyStart = Date()
            var av = audioND.mutableView(as: Float.self)
            av.withUnsafeMutablePointer { ptr, _, _ in
                audio.withUnsafeBufferPointer { src in
                    ptr.update(from: src.baseAddress!.advanced(by: chunkStart), count: realSamples)
                }
                if realSamples < chunkSamples {
                    ptr.advanced(by: realSamples).update(repeating: 0, count: chunkSamples - realSamples)
                }
            }
            let cpyMs = Date().timeIntervalSince(cpyStart) * 1000

            // 2. Preprocess (ANE).
            let preStart = Date()
            let mel = try await preprocess.extract(audio: audioND, length: lengthND)
            let preMs = Date().timeIntervalSince(preStart) * 1000

            // 3. Encoder (ANE).
            let encStart = Date()
            let encOut = try await encoder.forward(mel: mel, melLength: melLength)
            let encMs = Date().timeIntervalSince(encStart) * 1000

            return (encOut, cpyMs, preMs, encMs)
        }

        // Walk the audio in chunks. Maintain a "next chunk's encoder output"
        // task that runs concurrently with the current chunk's decoder.
        let totalChunks = (audio.count + chunkSamples - 1) / chunkSamples
        var currentEncOut: NDArray
        var currentTimings: (cpy: Double, pre: Double, enc: Double)
        do {
            // Chunk 0 runs synchronously to seed the pipeline.
            let chunkEnd = min(chunkSamples, audio.count)
            let realSamples = chunkEnd - 0
            let (encOut, cpyMs, preMs, encMs) = try await runUpToEncoder(
                chunkStart: 0, realSamples: realSamples
            )
            currentEncOut = encOut
            currentTimings = (cpyMs, preMs, encMs)
        }

        for chunkIdx in 0..<totalChunks {
            // Kick off chunk N+1's preprocess+encoder as a Task so it runs
            // concurrently with chunk N's decoder. Captured `audioND` is a
            // single shared buffer — the encoder doesn't retain it past its
            // own call, so it's safe to mutate after `runUpToEncoder` returns.
            let nextStart = (chunkIdx + 1) * chunkSamples
            let nextTask: Task<(NDArray, Double, Double, Double), Error>? = nextStart < audio.count
                ? Task {
                    let nextEnd = min(nextStart + chunkSamples, audio.count)
                    return try await runUpToEncoder(
                        chunkStart: nextStart, realSamples: nextEnd - nextStart
                    )
                }
                : nil

            // 4. Decoder for chunk N (Swift loop + many ANE joiner/predict calls).
            let decStart = Date()
            let newTokens = try await decoder.decode(encoderOutput: currentEncOut)
            perf.decodeMs += Date().timeIntervalSince(decStart) * 1000

            perf.copyMs += currentTimings.cpy
            perf.preprocessMs += currentTimings.pre
            perf.encodeMs += currentTimings.enc

            allEmittedTokens.append(contentsOf: newTokens)
            perf.chunkCount += 1

            // Wait for next chunk's encoder output.
            if let nextTask = nextTask {
                let (nextEncOut, cpyMs, preMs, encMs) = try await nextTask.value
                currentEncOut = nextEncOut
                currentTimings = (cpyMs, preMs, encMs)
            }
        }

        lastPerf = perf
        let fullText = await tokenizer.decode(tokens: allEmittedTokens, stripLangTags: true)
        let segment = Transcript.Segment(
            text: fullText,
            startTime: 0,
            endTime: Double(audio.count) / 16000.0
        )
        return Transcript(text: fullText, segments: [segment], language: language.rawValue)
    }

    public func finish() async throws -> Transcript {
        // If we have any leftover mel frames (< framesPerChunk), pad them
        // with zeros to a full chunk and run the encoder one last time.
        //
        // VERIFY: right-context padding for the final chunk. PLAN.md §2 says
        // the right-context frames are part of the chunk; for the final
        // chunk we won't have future audio, so we pad with zeros (silence).
        // Whether this matches NeMo's reference end-of-utterance handling is
        // an open question — Phase 1 / Phase 5 WER eval will surface any
        // tail-token drift caused by this padding. If NeMo's reference uses
        // a different padding convention (e.g., reflect, or right-context
        // truncation with a shorter final chunk), match it here.
        let framesPerChunk = Self.encoderFramesPerChunk
        if melBufferFrames > 0 && melBufferFrames < framesPerChunk {
            // Zero-pad the mel buffer in row-major order: each row m gets
            // (framesPerChunk - melBufferFrames) zeros appended after its
            // existing T frames.
            // Easiest to do by rebuilding the buffer with the new column count.
            var padded = [Float](repeating: 0, count: Self.nMels * framesPerChunk)
            for m in 0..<Self.nMels {
                for t in 0..<melBufferFrames {
                    padded[m * framesPerChunk + t] = melBuffer[m * melBufferFrames + t]
                }
                // The remaining columns [melBufferFrames..<framesPerChunk]
                // are already zero from the `repeating: 0` initializer.
            }
            melBuffer = padded
            melBufferFrames = framesPerChunk
        }

        // Drain whatever's left through the encoder. If melBufferFrames is
        // exactly 0, we skip — no audio left to process.
        if melBufferFrames >= framesPerChunk {
            let chunkMel = sliceMelChunk(frames: framesPerChunk)
            // Pass mel_length matching the chunk's frame count. The English-only
            // checkpoint takes (mel_chunk, mel_length); the multilingual variant
            // would also take a `lang_id` here.
            var melLength = NDArray(shape: [1], scalarType: .int32)
            melLength.setInt32(at: 0, to: Int32(framesPerChunk))
            let encOut = try await encoder.forward(mel: chunkMel, melLength: melLength)
            let newTokens = try await decoder.decode(encoderOutput: encOut)
            allEmittedTokens.append(contentsOf: newTokens)
        }

        // Detokenize the full utterance. We strip lang tags by default in
        // the final transcript since callers requesting `finish()` expect
        // the clean text; the language is surfaced separately in
        // `Transcript.language`.
        let fullText = await tokenizer.decode(tokens: allEmittedTokens, stripLangTags: true)

        // Build a single placeholder segment spanning [0, now]. Per-token
        // time-aligned segments are a Phase F follow-up — the public shape
        // is already fixed (Types.swift `Transcript.Segment`) so callers
        // don't break when we wire it in.
        //
        // VERIFY: segment time-alignment. PLAN.md §6 doesn't pin down how
        // RNN-T greedy decoding tracks per-token timing; the natural
        // approach is to track the (t, chunk) coordinate at which each
        // non-blank was emitted in `GreedyRNNTDecoder.decode(...)` and
        // surface that here. Out of Phase F scope; flag for Phase 7.
        let now = CFAbsoluteTimeGetCurrent()
        let segment = Transcript.Segment(
            text: fullText,
            startTime: 0,
            endTime: now - sessionStartTime
        )

        return Transcript(
            text: fullText,
            segments: [segment],
            language: language.rawValue
        )
    }

    // MARK: - Private helpers

    /// Slice the first `frames` time-frames out of `melBuffer` and return
    /// them as an `NDArray` of shape `[1, 128, frames]`, then drop those
    /// frames from `melBuffer`.
    ///
    /// `melBuffer` is row-major `[nMels × melBufferFrames]`. The encoder
    /// wants `[batch=1, nMels=128, T=frames]`, also row-major. The repack:
    ///
    ///     chunkMel[0, m, t] = melBuffer[m * melBufferFrames + t]   for t in 0..<frames
    ///
    /// After repack, we rebuild `melBuffer` keeping only the columns
    /// `frames..<melBufferFrames` (those become the new column 0..<remaining).
    ///
    /// Precondition: `melBufferFrames >= frames`.
    private func sliceMelChunk(frames: Int) -> NDArray {
        precondition(
            melBufferFrames >= frames,
            "sliceMelChunk(frames: \(frames)) called with only \(melBufferFrames) frames buffered"
        )

        // Resolves the Phase-F VERIFY: SDK 27.0 uses
        // `mutableView(as:).withUnsafeMutablePointer { ptr, shape, strides in ... }`.
        // The closure takes (ptr, shape, strides) instead of a flat buffer; for a
        // contiguous row-major NDArray the same `m * frames + t` index math works
        // off `ptr`.
        var chunkMel = NDArray(shape: [1, Self.nMels, frames], scalarType: .float32)
        var chunkMelView = chunkMel.mutableView(as: Float.self)
        chunkMelView.withUnsafeMutablePointer { ptr, _, _ in
            for m in 0..<Self.nMels {
                for t in 0..<frames {
                    // Destination index: [0, m, t] in row-major [1, nMels, frames]
                    //   = m * frames + t
                    // Source index: [m, t] in row-major [nMels, melBufferFrames]
                    //   = m * melBufferFrames + t
                    ptr[m * frames + t] = melBuffer[m * melBufferFrames + t]
                }
            }
        }

        // Drop the consumed frames from melBuffer. Rebuild as row-major
        // [nMels × (melBufferFrames - frames)].
        let remaining = melBufferFrames - frames
        if remaining > 0 {
            var newBuffer = [Float](repeating: 0, count: Self.nMels * remaining)
            for m in 0..<Self.nMels {
                for t in 0..<remaining {
                    newBuffer[m * remaining + t] = melBuffer[m * melBufferFrames + (frames + t)]
                }
            }
            melBuffer = newBuffer
            melBufferFrames = remaining
        } else {
            melBuffer.removeAll(keepingCapacity: true)
            melBufferFrames = 0
        }

        return chunkMel
    }
}
