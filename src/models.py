"""Chargement des VLMs en fp16 / 8 bits / 4 bits et génération de texte."""
import gc
import time
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLM

def load(model_id: str, mode: str = "fp16"):
    kwargs = {"device_map": "auto"}
    if mode == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif mode == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Mode inconnu : {mode}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    model.eval()
    return processor, model


def free(model):
    """Libère la VRAM entre deux configurations."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def generate(processor, model, image, prompt: str, max_new_tokens: int = 60):
    """Retourne (texte, latence en secondes). Décodage déterministe."""
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    latency = time.time() - t0
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip(), latency
