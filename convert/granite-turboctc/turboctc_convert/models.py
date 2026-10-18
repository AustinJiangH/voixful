"""torch.export-friendly wrapper for Granite Speech 5.0 TurboCTC.

One static graph per audio window:

  ctc_<F> : (audio [1, F*320] f32, length [1] i32) -> ids [1, F//4] i32
            — in-graph front-end (exact torchaudio MelSpectrogram via
            framing-gather + DFT matmuls, log10 floor, deltas, 2-frame
            stacking), the 16-layer block-attention conformer with
            self-conditioned CTC, the CTC head, and an IN-GRAPH argmax
            (avoids a [F/4, 16384] logits readback).

`length` is the number of real (unpadded) samples. It drives the encoder's
padding mask exactly like the upstream feature extractor's attention mask, so
a fixed-size window decodes the same as the unpadded clip; output positions
past the real length are forced to the blank id (0).

Host glue (Swift): CTC collapse (unique-consecutive, drop blank 0) and
byte-level BPE detokenize — no decode loop, no LLM.

Windows F must be multiples of 4 (two stride-2 subsampling blocks).
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
SAMPLES_PER_FRAME = 2 * HOP  # one stacked 50 Hz encoder-input frame = 320 samples
SUBSAMPLE = 4                # encoder output rate: 12.5 Hz
BLANK_ID = 0


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


class TurboCTCFrontEnd(nn.Module):
    """audio [1, F*320] -> features [1, F, 320] (stacked log-mel+delta pairs).

    GraniteSpeech5FeatureExtractor equivalence: mel power spectrum over 2F
    frames, log10(clamp 1e-10), floor at max - 8, /4 + 1, concat
    torchaudio.compute_deltas(win_length=3), stack 2 frames.
    """

    def __init__(self, frames: int):
        super().__init__()
        self.frames = frames
        num_samples = frames * SAMPLES_PER_FRAME
        mel_t = 2 * frames
        dft_real, dft_imag, fb = _mel_constants()
        self.register_buffer("dft_real", dft_real)
        self.register_buffer("dft_imag", dft_imag)
        self.register_buffer("fbanks", fb)
        self.register_buffer("pad_idx", _reflect_pad_indices(num_samples))
        starts = torch.arange(mel_t, dtype=torch.int64) * HOP
        offs = torch.arange(N_FFT, dtype=torch.int64)
        self.register_buffer("frame_idx", (starts.unsqueeze(1) + offs.unsqueeze(0)).reshape(-1))
        # compute_deltas(win_length=3) = (x[t+1] - x[t-1]) / 2 with replicate
        # padding at the edges, expressed as two gathers.
        t = torch.arange(mel_t, dtype=torch.int64)
        self.register_buffer("next_idx", torch.clamp(t + 1, max=mel_t - 1))
        self.register_buffer("prev_idx", torch.clamp(t - 1, min=0))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel_t = 2 * self.frames
        padded = audio[0].index_select(0, self.pad_idx)              # [S+512]
        frames = padded.index_select(0, self.frame_idx).view(mel_t, N_FFT)
        re = frames @ self.dft_real                                   # [2F, 257]
        im = frames @ self.dft_imag
        mel = (re * re + im * im) @ self.fbanks                       # [2F, 80]
        logmel = torch.log10(torch.clamp_min(mel, 1e-10))
        # global max as chained single-dim reductions: coreai_torch's amax
        # resolver requires explicit dims (argless amax -> IndexError)
        mx = logmel.amax(dim=1, keepdim=True).amax(dim=0, keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0) / 4.0 + 1.0
        delta = (logmel.index_select(0, self.next_idx) - logmel.index_select(0, self.prev_idx)) / 2.0
        feats = torch.cat((logmel, delta), dim=1)                     # [2F, 160]
        return feats.reshape(1, self.frames, 4 * N_MELS)              # [1, F, 320]


def _subsampling_forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
    """GraniteSpeech5EncoderSubsamplingBlock.forward with the 2x time pool
    written as reshape+mean instead of `unfold(1, 2, 2)` (aten.unfold has no
    coreai_torch lowering). Identical for the even lengths every window here
    produces (F is a multiple of 4)."""
    residual = hidden_states
    hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
    hidden_states = residual + 0.5 * hidden_states
    attn_output, _ = self.self_attn(
        hidden_states=self.norm_self_att(hidden_states),
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = hidden_states + attn_output
    conv_output = self.conv(self.norm_conv(hidden_states), attention_mask=attention_mask)
    b, t, c = hidden_states.shape
    pooled = hidden_states[:, : 2 * (t // 2)].reshape(b, t // 2, 2, c).mean(dim=2)
    hidden_states = pooled + conv_output[:, : pooled.shape[1]]
    hidden_states = hidden_states + 0.5 * self.feed_forward2(self.norm_feed_forward2(hidden_states))
    return self.norm_out(hidden_states)


def make_exportable(encoder: nn.Module) -> None:
    """Swap the subsampling blocks' forward for the export-clean version."""
    import types

    for layer in encoder.layers:
        if type(layer).__name__.endswith("SubsamplingBlock"):
            layer.forward = types.MethodType(_subsampling_forward, layer)


class TurboCTCForCoreAI(nn.Module):
    """(audio [1, F*320], length [1] i32) -> ids [1, F//4] i32."""

    def __init__(self, model: nn.Module, frames: int):
        super().__init__()
        assert frames % SUBSAMPLE == 0, f"window {frames} not a multiple of {SUBSAMPLE}"
        self.frames = frames
        self.mel = TurboCTCFrontEnd(frames)
        self.encoder = model.encoder
        make_exportable(self.encoder)
        self.ctc_head = model.ctc_head
        self.register_buffer("frame_pos", torch.arange(frames, dtype=torch.int32).unsqueeze(0))
        self.register_buffer(
            "out_pos", torch.arange(frames // SUBSAMPLE, dtype=torch.int32).unsqueeze(0)
        )

    def forward(self, audio: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        feats = self.mel(audio)
        # Upstream mask: ceil((samples // hop) / 2) valid stacked frames;
        # the encoder halves it per subsampling block (floor), so the valid
        # output length is n_feat // 4.
        mel_frames = torch.div(length, HOP, rounding_mode="floor")
        n_feat = torch.div(mel_frames + 1, 2, rounding_mode="floor")
        mask = self.frame_pos < n_feat.view(1, 1)                     # [1, F] bool
        hidden = self.encoder(feats, attention_mask=mask).last_hidden_state
        ids = self.ctc_head(hidden).argmax(dim=-1).to(torch.int32)    # [1, F/4]
        n_out = torch.div(n_feat, SUBSAMPLE, rounding_mode="floor")
        valid = self.out_pos < n_out.view(1, 1)
        return torch.where(valid, ids, torch.zeros_like(ids))
