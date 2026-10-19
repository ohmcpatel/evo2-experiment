# Evo 2 sequence comparison

A Streamlit experiment for comparing wild-type and mutant DNA sequence likelihoods with NVIDIA-hosted Evo 2.

The app shows aligned sequences, per-position log-probabilities, score differences, and nucleotide composition. A model likelihood difference is not a validated measure of biological fitness, pathogenicity, or molecular stability.

## Run locally

```bash
git clone https://github.com/ohmcpatel/evo2-experiment.git
cd evo2-experiment/wt_mutant_fitness
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set your own `EVO2_API_KEY` in `.env`, then run:

```bash
streamlit run app.py
```

Open `http://localhost:8501`. API access and endpoint availability are required; requests use the NVIDIA service and may incur provider usage charges.

## Files

- [`app.py`](wt_mutant_fitness/app.py) — Streamlit interface.
- [`api_client.py`](wt_mutant_fitness/api_client.py) — Hosted model requests.
- [`scoring.py`](wt_mutant_fitness/scoring.py) — Sequence scoring.
- [Detailed notes](wt_mutant_fitness/README.md) — Inputs and interpretation.

This is an exploratory interface, with no variant-specific validation reported here. Keep API keys in your local environment; `.env` and Python caches are ignored.
