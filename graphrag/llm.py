import numpy as np
import ollama

from .config import EMBED_MODEL, LLM_MODEL


def local_chat(prompt, json_mode=False):
    kwargs = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0},
    }
    if json_mode:
        kwargs["format"] = "json"

    response = ollama.chat(**kwargs)
    return response["message"]["content"]


def get_embedding(text):
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return np.asarray(response["embeddings"][0], dtype=np.float32)
