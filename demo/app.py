"""Démo Gradio : image -> légende ou score, avec choix modèle + précision.
Lancement (Colab) : python demo/app.py
"""
import sys, yaml
sys.path.insert(0, ".")
import torch
import gradio as gr
from src.models import load, free, generate
from src.metrics import peak_vram_gb

cfg = yaml.safe_load(open("configs/default.yaml"))
MODELS = {m["name"]: m["hf_id"] for m in cfg["models"]}
_state = {"key": None, "proc": None, "model": None}


def run(image, model_name, precision, task, caption_to_judge):
    key = (model_name, precision)
    if _state["key"] != key:
        if _state["model"] is not None:
            free(_state["model"])
        _state["proc"], _state["model"] = load(MODELS[model_name], precision)
        _state["key"] = key
    torch.cuda.reset_peak_memory_stats()
    prompt = (cfg["prompts"]["caption"] if task == "Légende"
              else cfg["prompts"]["judge"].format(caption=caption_to_judge))
    text, lat = generate(_state["proc"], _state["model"], image, prompt,
                         cfg["generation"]["max_new_tokens"])
    return text, f"VRAM max : {peak_vram_gb():.2f} Go | latence : {lat:.2f} s"


demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(type="pil"),
            gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Modèle"),
            gr.Dropdown(["fp16", "int8", "int4"], value="fp16", label="Précision"),
            gr.Radio(["Légende", "Score"], value="Légende", label="Tâche"),
            gr.Textbox(label="Légende à noter (mode Score)")],
    outputs=[gr.Textbox(label="Résultat"), gr.Textbox(label="Ressources")],
    title="Petits VLMs : légende ou jugement, selon la quantification",
)

if __name__ == "__main__":
    demo.launch(share=True)
