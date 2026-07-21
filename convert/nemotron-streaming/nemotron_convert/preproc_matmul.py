"""Export-friendly audio->mel preprocessor for the Nemotron batch path.

NeMo's AudioToMelSpectrogramPreprocessor doesn't survive torch.export
(dynamic masking via make_seq_mask_like), so this module re-expresses the
same math with fixed shapes and matmuls only:

  preemph(0.97) -> frame [512]@hop160 (torch.stft center=True equivalence
  via the 512-sample REAL left context in the input buffer) -> centered
  400-Hann -> DFT as two [512,257] matmuls -> power -> Slaney mel fb ->
  log(x + 2^-24)

Input  : audio [1, 20832]  (512 left-context samples + 20320 chunk samples)
         audio_length [1]  (accepted for interface parity; unused — shapes
                            are fixed and the caller zero-pads short tails)
Output : mel [1, 128, 128] (mel bins x frames), matching PreprocessSession.

Frame k (0..127) covers preemphasized buffer [k*160+256, k*160+768) — i.e.
torch.stft(center=True) frame k of the CHUNK, with the reflect-pad replaced
by real previous-chunk audio on the left and zeros past the buffer end on
the right (only the last two frames' tails; mid-utterance chunks get real
right context from the next chunk's overlap being in ITS left context).

All constants (window, mel filterbank, preemph, log guard) are pulled from
the loaded NeMo featurizer at construction, so parity is by construction —
verified by `python -m nemotron_convert.preproc_matmul` (self-test).
"""

from __future__ import annotations

import math

import torch
from torch import nn

AUDIO_SAMPLES = 20832
LEFT_CONTEXT = 512
CHUNK_SAMPLES = 20320
N_FFT = 512
HOP = 160
WIN = 400
N_MELS = 128
OUT_FRAMES = 128


class NemotronPreprocessorForCoreAI(nn.Module):
    def __init__(self, featurizer: nn.Module):
        super().__init__()
        self.preemph = float(featurizer.preemph)

        hann = featurizer.window.detach().float()           # [400]
        assert hann.numel() == WIN, f"window {hann.numel()} != {WIN}"
        w = torch.zeros(N_FFT)
        pad = (N_FFT - WIN) // 2
        w[pad : pad + WIN] = hann                            # centered, stft win<n_fft
        self.register_buffer("window", w)

        fb = featurizer.fb.detach().float().squeeze()        # -> [257, 128] or [128, 257]
        if fb.shape[0] == N_MELS:
            fb = fb.T
        assert fb.shape == (N_FFT // 2 + 1, N_MELS), f"fb {tuple(fb.shape)}"
        self.register_buffer("melfb", fb)

        k = torch.arange(N_FFT, dtype=torch.float64)
        n = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)
        ang = -2.0 * math.pi * torch.outer(k, n) / N_FFT
        self.register_buffer("dft_re", torch.cos(ang).float())   # [512, 257]
        self.register_buffer("dft_im", torch.sin(ang).float())

        guard = getattr(featurizer, "log_zero_guard_value", 2.0 ** -24)
        self.guard = float(guard)

        # Constant gather indices for framing — coreai_torch has no lowering
        # for aten.unfold, so frames are gathered by index instead: frame k
        # covers padded-buffer [k*160+256, k*160+768).
        starts = torch.arange(OUT_FRAMES) * HOP + 256
        idx = (starts.view(-1, 1) + torch.arange(N_FFT).view(1, -1)).reshape(-1)
        self.register_buffer("frame_idx", idx)  # [128*512] int64

    def forward(self, audio: torch.Tensor, audio_length: torch.Tensor) -> torch.Tensor:
        x = audio  # [1, 20832]
        y = torch.cat([x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]], dim=1)
        # Right-pad so frame 127's 512-sample window fits; frames start at
        # k*160 + 256 (stft center equivalence given the 512 left context).
        y = torch.nn.functional.pad(y, (0, 256))
        frames = y[:, self.frame_idx].view(1, OUT_FRAMES, N_FFT)  # [1, 128, 512]
        frames = frames * self.window
        re = frames @ self.dft_re                            # [1, 128, 257]
        im = frames @ self.dft_im
        power = re * re + im * im
        mel = power @ self.melfb                             # [1, 128fr, 128mel]
        mel = torch.log(mel + self.guard)
        return mel.transpose(1, 2)                           # [1, 128mel, 128fr]


def _self_test() -> int:
    """Parity vs NeMo's preprocessor on random audio: mid frames must match
    tightly; the last ~2 frames may differ (zero vs reflect right-pad)."""
    import nemo.collections.asr as nemo_asr

    m = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b"
    )
    feat = m.preprocessor.featurizer
    feat.dither = 0.0
    mod = NemotronPreprocessorForCoreAI(feat).eval()

    torch.manual_seed(0)
    utt = torch.randn(1, LEFT_CONTEXT + CHUNK_SAMPLES) * 0.1

    # Reference: NeMo over the CHUNK ONLY (center=True reflect-pads its start;
    # our version uses the real left context, so compare from frame 4 on where
    # the 512-sample histories coincide).
    with torch.no_grad():
        ref, _ = m.preprocessor(
            input_signal=utt[:, LEFT_CONTEXT:],
            length=torch.tensor([CHUNK_SAMPLES]),
        )
        got = mod(utt, torch.tensor([AUDIO_SAMPLES], dtype=torch.int32))

    n = min(ref.shape[2], got.shape[2])
    mid = slice(4, n - 2)
    err = (ref[:, :, mid] - got[:, :, mid]).abs().max().item()
    print(f"mid-frame max|Δ| = {err:.6e} over frames {mid} (ref {ref.shape}, got {got.shape})")
    ok = err < 1e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
