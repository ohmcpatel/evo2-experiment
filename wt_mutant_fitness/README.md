# Evo 2 · Variant Fitness Lab

A scientific dashboard for comparing wild-type vs mutant DNA fitness using the NVIDIA-hosted Evo 2 40B model.

## Quick start

```bash
cd experiments/wt_mutant_fitness

# Install dependencies
pip install -r requirements.txt

# Set your NVIDIA API key
cp .env.example .env
# Edit .env and paste your key: EVO2_API_KEY=nvapi-...

# Launch the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## What it does

1. You paste a **wild-type** and **mutant** DNA sequence (A/C/G/T only).
2. The app calls the **Evo 2 40B forward endpoint** on NVIDIA's hosted API to get logits for each sequence.
3. It converts logits to per-position log-probabilities (same math as `evo2/scoring.py` in the main repo) and computes a scalar fitness score (mean log P).
4. **ΔScore = Mutant − WT**: negative means destabilizing, positive means stabilizing.
5. Three visualization tabs:
   - **Strand View**: both sequences side-by-side with mutations highlighted in color.
   - **Log-Probability**: interactive Plotly chart of per-position log P for both sequences and their difference.
   - **Composition**: nucleotide composition and GC% comparison.

## API key

Get a free key at [build.nvidia.com/arc/evo2-40b](https://build.nvidia.com/arc/evo2-40b). Store it in `.env` as `EVO2_API_KEY`. Never commit `.env`.

## Notes

- The forward endpoint (`/biology/arc/evo2-40b/forward`) returns logits for the full input sequence. If the hosted API does not expose this endpoint, you will see an error — in that case, a self-hosted NIM instance is needed.
- For very long sequences (>10 kbp), API latency may be significant.
- Scores are zero-shot: no variant-specific training is used.
