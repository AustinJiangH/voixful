"""torch.export-friendly wrappers for Granite Speech 4.1 2B NAR.

The HF forward() is not export-clean (``.tolist()`` on lengths, pad-free
flattening, per-sample Python loops), so the model is split into two static
graphs plus microsecond-scale host glue:

  encode_<F>  : raw audio [1, F*320] -> (audio_embeds [1, F//5, 2048],
                ctc_ids [1, F//4] int32)
                — in-graph mel front-end (exact torchaudio MelSpectrogram
                equivalence via framing-gather + DFT matmuls), 16-layer
                conformer with self-conditioning, posterior-weighted pool,
                BPE CTC head with IN-GRAPH argmax (avoids a [F/4, 100352]
                logits readback), multi-layer Q-Former projector, and the
                /embedding_multiplier scaling from the reference forward.
  edit_<A>_<U>: (audio_embeds [1, A, 2048], text_ids [1, U] int32,
                attn_mask [1, A+U]) -> token ids [1, U] int32
                — ONE bidirectional pass of the 40-layer Granite LLM over
                [audio ; interleaved-CTC-hypothesis], lm_head only over the
                text segment, in-graph argmax.

Host glue (Swift): CTC collapse (argmax ids -> unique_consecutive -> drop
blank), insertion-slot interleave (blank,t1,blank,...,blank; min length 8),
pad text_ids to U with blank + 0-mask (masked keys are invisible to real
positions, so bucket padding is parity-exact), final CTC collapse.

Design precondition (verified by granite_convert.parity): padding AUDIO with
silence to a fixed window leaves transcripts unchanged, so all shapes are
static and the encoder's missing internal padding mask is irrelevant.

Windows F must be multiples of 200 (the encoder's attention block) — which
also makes them multiples of the pooling window 4 — and of 5 (projector
downsample). Multiples of 200 satisfy all three.
"""

from __future__ import annotations

import math

import torch
from torch import nn

SAMPLE_RATE = 16000
HOP = 160
N_FFT = 512
WIN_LENGTH = 400
N_MELS = 80
SAMPLES_PER_FRAME = 2 * HOP  # one 50 Hz encoder frame = 320 samples


def _mel_constants() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dft_real [512, 257], dft_imag [512, 257], fbanks [257, 80]) with the
    Hann window folded into the DFT matrices. Matches torchaudio
    MelSpectrogram(power=2, mel_scale='htk', norm=None) exactly."""
    import torchaudio

    window = torch.zeros(N_FFT, dtype=torch.float64)
    pad = (N_FFT - WIN_LENGTH) // 2  # torch.stft centers win_length in n_fft
    window[pad : pad + WIN_LENGTH] = torch.hann_window(WIN_LENGTH, periodic=True, dtype=torch.float64)

    n_freqs = N_FFT // 2 + 1
    j = torch.arange(N_FFT, dtype=torch.float64).unsqueeze(1)
    k = torch.arange(n_freqs, dtype=torch.float64).unsqueeze(0)
    ang = 2.0 * math.pi * j * k / N_FFT
    dft_real = (torch.cos(ang) * window.unsqueeze(1)).to(torch.float32)
    dft_imag = (-torch.sin(ang) * window.unsqueeze(1)).to(torch.float32)

    fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs, f_min=0.0, f_max=SAMPLE_RATE / 2, n_mels=N_MELS,
        sample_rate=SAMPLE_RATE, norm=None, mel_scale="htk",
    ).to(torch.float32)  # [257, 80]
    return dft_real, dft_imag, fb


def _reflect_pad_indices(num_samples: int) -> torch.Tensor:
    """Index map implementing torch.stft's center-mode reflect padding
    (n_fft//2 on each side) as a gather — aten.pad reflect + unfold are
    not exportable through coreai_torch."""
    half = N_FFT // 2
    idx = torch.empty(num_samples + N_FFT, dtype=torch.int64)
    for p in range(num_samples + N_FFT):
        s = p - half
        if s < 0:
            s = -s
        elif s >= num_samples:
            s = 2 * num_samples - 2 - s
        idx[p] = s
    return idx


class GraniteMelFrontEnd(nn.Module):
    """audio [1, F*320] -> features [1, F, 160] (stacked log-mel pairs).

    Exact GraniteSpeechNarFeatureExtractor equivalence: mel power spectrum,
    truncate to 2F frames, log10(clamp 1e-10), clip to global max - 8,
    /4 + 1, stack 2 frames.
    """

    def __init__(self, frames: int):
        super().__init__()
        self.frames = frames
        num_samples = frames * SAMPLES_PER_FRAME
        mel_t = 2 * frames  # truncated frame count l = 2*(S // 320)
        dft_real, dft_imag, fb = _mel_constants()
        self.register_buffer("dft_real", dft_real)
        self.register_buffer("dft_imag", dft_imag)
        self.register_buffer("fbanks", fb)
        self.register_buffer("pad_idx", _reflect_pad_indices(num_samples))
        # frame_idx[t, j] = start-of-frame t (t*160) + j into the padded signal
        starts = torch.arange(mel_t, dtype=torch.int64) * HOP
        offs = torch.arange(N_FFT, dtype=torch.int64)
        self.register_buffer("frame_idx", (starts.unsqueeze(1) + offs.unsqueeze(0)).reshape(-1))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel_t = 2 * self.frames
        padded = audio[0].index_select(0, self.pad_idx)             # [S+512]
        frames = padded.index_select(0, self.frame_idx).view(mel_t, N_FFT)
        re = frames @ self.dft_real                                  # [2F, 257]
        im = frames @ self.dft_imag
        power = re * re + im * im
        mel = power @ self.fbanks                                    # [2F, 80]
        logmel = torch.log10(torch.clamp_min(mel, 1e-10))
        # global max as chained single-dim reductions: coreai_torch's amax
        # resolver requires explicit dims (argless amax -> IndexError)
        mx = logmel.amax(dim=1, keepdim=True).amax(dim=0, keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0) / 4.0 + 1.0
        return logmel.reshape(1, self.frames, 2 * N_MELS)            # [1, F, 160]


class GraniteEncoderForCoreAI(nn.Module):
    """audio [1, F*320] -> (audio_embeds [1, F//5, 2048], ctc_ids [1, F//4] i32).

    Replicates GraniteSpeechNarCTCEncoder.forward + the projector/scaling part
    of GraniteSpeechNarForASR.forward, minus the data-dependent flattening
    (batch 1, full window: lengths are compile-time constants).
    """

    def __init__(self, asr_model: nn.Module, frames: int):
        super().__init__()
        assert frames % 200 == 0, "windows must be multiples of the 200-frame attention block"
        self.frames = frames
        self.mel = GraniteMelFrontEnd(frames)
        self.encoder = asr_model.encoder
        self.projector = asr_model.projector
        enc_cfg = asr_model.encoder.config
        self.self_conditioning_layer = enc_cfg.self_conditioning_layer
        self.pool_window = enc_cfg.bpe_pooling_window          # 4
        self.layer_indices = list(asr_model.config.encoder_layer_indices)  # [4, 8, 12, -1]
        self.downsample = asr_model.projector.config.downsample_rate       # 5
        self.emb_mult = float(asr_model.config.text_config.embedding_multiplier)  # 12.0
        # Shaw relative-position index table (constant; mirrors encoder.forward)
        context = enc_cfg.context_size
        seq = torch.arange(context)
        dist = torch.clamp(seq.view(-1, 1) - seq.view(1, -1), -context, context) + enc_cfg.max_pos_emb
        self.register_buffer("attention_dists", dist)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder
        feats = self.mel(audio)
        hidden = enc.input_linear(feats)
        collected = [hidden]
        blank_probs = None
        for layer_idx, layer in enumerate(enc.layers, start=1):
            hidden = layer(hidden, attention_dists=self.attention_dists)
            if layer_idx == self.self_conditioning_layer:
                mid_probs = torch.softmax(enc.out(hidden).float(), dim=-1)
                blank_probs = mid_probs[:, :, 0]
                hidden = hidden + enc.out_mid(mid_probs.to(hidden.dtype))
            collected.append(hidden)

        # Posterior-weighted pooling (window 4; frames % 4 == 0, no pad branch)
        w = self.pool_window
        num_windows = self.frames // w
        importance = (1.0 - blank_probs).view(1, num_windows, w)
        weights = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
        pooled = (hidden.float().view(1, num_windows, w, -1) * weights.unsqueeze(-1)).sum(dim=2)
        bpe_logits = enc.out_bpe(pooled.to(hidden.dtype))            # [1, F/4, 100352]
        ctc_ids = torch.argmax(bpe_logits, dim=-1).to(torch.int32)   # [1, F/4]

        multilayer = torch.cat([collected[i] for i in self.layer_indices], dim=-1)
        audio_embeds = self.projector(multilayer) / self.emb_mult
        audio_embeds = audio_embeds[:, : self.frames // self.downsample]
        return audio_embeds, ctc_ids


class GraniteEditForCoreAI(nn.Module):
    """One bidirectional LLM pass over [audio ; slotted hypothesis].

    (audio_embeds [1, A, 2048], text_ids [1, U] i32, attn_mask [1, A+U])
      -> ids [1, U] i32

    attn_mask is 1 for valid positions, 0 for bucket padding; padded KEYS get
    an additive -inf so real positions compute reference-exact attention.
    lm_head runs only on the text segment (saves an [A, 100352] matmul).
    """

    def __init__(self, asr_model: nn.Module, audio_len: int, text_len: int):
        super().__init__()
        self.lm = asr_model.language_model
        self.audio_len = audio_len
        self.text_len = text_len
        self.emb_mult = float(asr_model.config.text_config.embedding_multiplier)
        self.logits_scaling = float(asr_model.config.text_config.logits_scaling)

    def forward(
        self, audio_embeds: torch.Tensor, text_ids: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        model = self.lm.model
        text_emb = model.embed_tokens(text_ids)
        hidden = torch.cat([audio_embeds, text_emb], dim=1) * self.emb_mult
        seq_len = self.audio_len + self.text_len
        position_ids = torch.arange(seq_len).unsqueeze(0)
        position_embeddings = model.rotary_emb(hidden, position_ids=position_ids)
        # Additive bidirectional mask, broadcast over queries: [1, 1, 1, L]
        neg = torch.finfo(hidden.dtype).min
        mask4d = ((1.0 - attn_mask) * neg).view(1, 1, 1, seq_len)
        for layer in model.layers:
            hidden = layer(
                hidden,
                attention_mask=mask4d,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        hidden = model.norm(hidden)
        text_hidden = hidden[:, self.audio_len :, :]
        logits = self.lm.lm_head(text_hidden) / self.logits_scaling
        return torch.argmax(logits, dim=-1).to(torch.int32)
