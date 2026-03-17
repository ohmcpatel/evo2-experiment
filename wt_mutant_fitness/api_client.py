import base64
import io
import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

API_BASE = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b"
FORWARD_URL = f"{API_BASE}/forward"
GENERATE_URL = f"{API_BASE}/generate"

NO_PROXY = {"http": None, "https": None}


def _headers():
    key = os.environ.get("EVO2_API_KEY") or os.environ.get("NVCF_RUN_KEY")
    if not key:
        raise RuntimeError(
            "Set EVO2_API_KEY (or NVCF_RUN_KEY) in your environment or .env file."
        )
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def forward(sequence: str) -> dict[str, np.ndarray]:
    """Call the Evo 2 forward endpoint and return logits.

    Returns a dict with 'logits' key -> numpy array of shape [seq_len, 512].
    Uses the 'unembed' layer which returns full-sequence logits on the hosted API.
    """
    payload = {"sequence": sequence, "output_layers": ["unembed"]}
    resp = requests.post(
        FORWARD_URL, headers=_headers(), json=payload,
        timeout=300, proxies=NO_PROXY,
    )
    resp.raise_for_status()

    body = resp.json()
    npz_b64 = body["data"]
    npz_bytes = base64.b64decode(npz_b64)
    npz = np.load(io.BytesIO(npz_bytes))

    raw = npz["unembed.output"]  # shape (1, seq_len, 512)
    logits = raw[0] if raw.ndim == 3 else raw  # -> (seq_len, 512)

    return {"logits": logits}


def generate(
    sequence: str,
    num_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 3,
) -> dict:
    """Call the Evo 2 generate endpoint. Returns the full JSON response."""
    payload = {
        "sequence": sequence,
        "num_tokens": num_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "enable_sampled_probs": True,
        "enable_logits": True,
    }
    resp = requests.post(
        GENERATE_URL, headers=_headers(), json=payload,
        timeout=300, proxies=NO_PROXY,
    )
    resp.raise_for_status()
    return resp.json()
