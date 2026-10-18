"""Load the upstream Granite Speech 5.0 TurboCTC checkpoint (native transformers).

The checkpoint's weights use transformers' native GraniteSpeech5ForCTC names
(transformers>=5.16); the legacy modeling .py files still in the repo no longer
match them, so only the native classes are used here.
"""

from __future__ import annotations

MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"


def load_model(model_id: str = MODEL_ID):
    """-> (processor, model) in fp32 eval mode on CPU."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id, dtype=torch.float32).eval()
    return processor, model


def reference_transcribe(processor, model, audio) -> str:
    """Upstream pipeline end to end on unpadded audio: the parity target."""
    import torch

    with torch.no_grad():
        inputs = processor([audio], sampling_rate=16000)
        ids = model.generate(**inputs)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]
