# Wild-type / mutant sequence comparison

A Streamlit interface for comparing DNA sequence likelihoods from Evo 2.

## Setup

From the repository root:

```bash
cd wt_mutant_fitness
python -m pip install -r requirements.txt
cp .env.example .env
# Set your own EVO2_API_KEY in .env.
streamlit run app.py
```

See the [root README](../README.md) for the full setup. The app opens at `http://localhost:8501` and requires access to the configured NVIDIA-hosted API.

## What it computes

1. Supply wild-type and mutant DNA sequences using A/C/G/T.
2. The client requests model logits for each sequence.
3. The scorer converts those logits to per-position log-probabilities and a mean sequence score.
4. The app compares mutant minus wild-type scores and displays alignments, probability plots, and nucleotide composition.

A positive difference means the mutant receives a higher score under this model; a negative difference means a lower score. Neither direction establishes biological fitness, pathogenicity, or molecular stability. The interface's historical “fitness” labels should be read with that limitation.

Endpoint access, response format, and practical sequence-length limits depend on the provider. No external biological validation is reported in this repository.
