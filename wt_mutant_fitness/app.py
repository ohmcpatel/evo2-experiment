"""
Evo 2 Variant Fitness Lab
─────────────────────────
A scientific dashboard for comparing wild-type vs mutant DNA fitness
using the NVIDIA Evo 2 40B foundation model.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import textwrap

from api_client import forward, generate
from scoring import (
    tokenize,
    extract_logits,
    score_sequence,
    per_position_scores,
    find_mutations,
    reverse_complement,
    BASE_TO_ASCII,
    ASCII_TO_BASE,
)

# ──────────────────────────────────────────────
# Page config & custom CSS
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Evo 2 · Variant Fitness Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --accent-cyan: #06d6a0;
    --accent-magenta: #ef476f;
    --accent-amber: #ffd166;
    --accent-blue: #118ab2;
    --text-primary: #e8edf5;
    --text-secondary: #8892a4;
    --border-subtle: #1e293b;
    --base-a: #06d6a0;
    --base-t: #ef476f;
    --base-c: #118ab2;
    --base-g: #ffd166;
}

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1400px;
}

h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-secondary);
    font-size: 0.85rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 1.6rem !important;
}

/* Text areas */
textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    background: #0d1220 !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #06d6a0, #118ab2) !important;
    color: #0a0e17 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(6, 214, 160, 0.3);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary) !important;
    border-bottom: 2px solid transparent;
    padding: 0.8rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

/* Divider */
hr {
    border-color: var(--border-subtle) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--accent-cyan) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown(
    """
    <div style="margin-bottom: 0.5rem;">
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
              color:#8892a4; text-transform:uppercase; letter-spacing:0.12em;">
            Arc Institute · Evo 2 40B
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("Variant Fitness Lab")
st.markdown(
    '<p style="color:#8892a4; margin-top:-0.8rem; font-size:1.05rem;">'
    "Compare wild-type and mutant DNA fitness using Evo 2's sequence likelihood scoring."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.8rem 0;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:1rem;
                  font-weight:700; color:#e8edf5; letter-spacing:-0.02em;">
                ⚙ Configuration
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scoring_method = st.selectbox(
        "Scoring method",
        ["mean", "sum"],
        help="'mean' normalizes by length; 'sum' gives total log-likelihood (favors longer seqs)",
    )

    window_display = st.slider(
        "Strand view window (bp)",
        min_value=30,
        max_value=200,
        value=80,
        step=10,
        help="Number of bases shown around the mutation in the strand viewer",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="padding:0.4rem 0;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                  color:#8892a4; text-transform:uppercase; letter-spacing:0.08em;">
                About
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Scores are **mean per-position log-likelihoods** from Evo 2's forward pass. "
        "**ΔScore = Mutant − Wild-type**. Negative values suggest the mutation is "
        "destabilizing (lower fitness). This is a zero-shot metric — no variant-specific "
        "training is used.",
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.7rem; color:#555e6e;">'
        "Powered by NVIDIA NIM · Evo 2 40B · Arc Institute"
        "</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Example sequences (BRCA1 region, short demo)
# ──────────────────────────────────────────────

EXAMPLE_WT = (
    "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCA"
    "TCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACT"
    "TCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAGCCTACAAGAAAGTACGAG"
)

EXAMPLE_MUT = (
    "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCA"
    "TCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACT"
    "TCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAATGGAGCCTACAAGAAAGTACGAG"
)


# ──────────────────────────────────────────────
# Input section
# ──────────────────────────────────────────────

col_wt, col_mut = st.columns(2, gap="large")

with col_wt:
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.75rem; '
        'color:#06d6a0; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;">'
        "Wild-Type Sequence</p>",
        unsafe_allow_html=True,
    )
    wt_input = st.text_area(
        "Wild-type",
        value=EXAMPLE_WT,
        height=140,
        label_visibility="collapsed",
        placeholder="Paste wild-type DNA sequence (A/C/G/T only)…",
    )

with col_mut:
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.75rem; '
        'color:#ef476f; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;">'
        "Mutant Sequence</p>",
        unsafe_allow_html=True,
    )
    mut_input = st.text_area(
        "Mutant",
        value=EXAMPLE_MUT,
        height=140,
        label_visibility="collapsed",
        placeholder="Paste mutant DNA sequence (A/C/G/T only)…",
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def render_strand_html(wt_seq: str, mut_seq: str, mutations: list[dict], window: int) -> str:
    """Produce an HTML block showing both strands with mutations highlighted."""
    if not mutations:
        return '<p style="color:#8892a4;">No mutations detected.</p>'

    center = mutations[0]["pos"]
    half = window // 2
    start = max(0, center - half)
    end = min(len(wt_seq), center + half)

    base_colors = {"A": "#06d6a0", "T": "#ef476f", "C": "#118ab2", "G": "#ffd166"}
    mut_positions = {m["pos"] for m in mutations}

    def build_row(seq, label, label_color, is_mut_row=False):
        chars = []
        for i in range(start, end):
            ch = seq[i] if i < len(seq) else " "
            color = base_colors.get(ch.upper(), "#8892a4")
            if i in mut_positions:
                bg = "rgba(239,71,111,0.25)" if is_mut_row else "rgba(6,214,160,0.15)"
                border_c = "#ef476f" if is_mut_row else "#06d6a0"
                chars.append(
                    f'<span style="background:{bg}; border-bottom:2px solid {border_c}; '
                    f'color:{color}; padding:0 1px; border-radius:2px; font-weight:700;">{ch}</span>'
                )
            else:
                chars.append(f'<span style="color:{color};">{ch}</span>')

        pos_label = f'<span style="color:#555e6e; font-size:0.65rem;">{start}</span>'
        end_label = f'<span style="color:#555e6e; font-size:0.65rem;">{end}</span>'
        label_span = (
            f'<span style="color:{label_color}; font-size:0.65rem; text-transform:uppercase; '
            f'letter-spacing:0.08em; font-weight:600; display:inline-block; width:36px;">{label}</span>'
        )

        return (
            f'<div style="margin:2px 0;">'
            f'{label_span} {pos_label} {"".join(chars)} {end_label}'
            f'</div>'
        )

    position_markers = []
    for i in range(start, end):
        if i in mut_positions:
            position_markers.append(
                '<span style="color:#ef476f; font-weight:700;">▼</span>'
            )
        else:
            position_markers.append('<span style="color:transparent;">·</span>')
    marker_pad = '<span style="display:inline-block; width:36px;"></span> '
    marker_pad += '<span style="font-size:0.65rem; color:transparent;">000</span> '
    marker_row = f'<div style="margin:0;">{marker_pad}{"".join(position_markers)}</div>'

    mutation_info = " · ".join(
        f'<span style="color:#ffd166; font-weight:600;">pos {m["pos"]}</span> '
        f'<span style="color:#06d6a0;">{m["ref"]}</span>'
        f'<span style="color:#555e6e;">→</span>'
        f'<span style="color:#ef476f;">{m["alt"]}</span>'
        for m in mutations
    )

    return (
        f'<div style="background:#0d1220; border:1px solid #1e293b; border-radius:10px; '
        f'padding:1.2rem 1.4rem; font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; '
        f'line-height:1.9; overflow-x:auto;">'
        f'<div style="margin-bottom:0.7rem; font-size:0.7rem; color:#8892a4;">'
        f'MUTATIONS: {mutation_info}</div>'
        f'{marker_row}'
        f'{build_row(wt_seq, "WT", "#06d6a0", False)}'
        f'{build_row(mut_seq, "MUT", "#ef476f", True)}'
        f'<div style="margin-top:0.5rem; font-size:0.6rem; color:#555e6e;">'
        f'Showing positions {start}–{end} of {len(wt_seq)} bp'
        f'</div>'
        f'</div>'
    )


def build_logprob_chart(
    wt_lp: np.ndarray, mut_lp: np.ndarray, mutations: list[dict]
) -> go.Figure:
    """Interactive Plotly chart of per-position log-probabilities."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=["Per-position log P(next base)", "ΔLog P  (Mutant − WT)"],
    )

    x = np.arange(len(wt_lp))
    delta = mut_lp - wt_lp

    fig.add_trace(
        go.Scatter(
            x=x, y=wt_lp, name="Wild-type",
            line=dict(color="#06d6a0", width=1.2),
            opacity=0.85,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=mut_lp, name="Mutant",
            line=dict(color="#ef476f", width=1.2),
            opacity=0.85,
        ),
        row=1, col=1,
    )

    colors = ["#ef476f" if d < 0 else "#06d6a0" for d in delta]
    fig.add_trace(
        go.Bar(
            x=x, y=delta, name="Δ",
            marker_color=colors, opacity=0.7,
            showlegend=False,
        ),
        row=2, col=1,
    )

    for m in mutations:
        pos = m["pos"]
        if pos < len(wt_lp):
            fig.add_vline(
                x=pos, line_width=1, line_dash="dot",
                line_color="#ffd166", opacity=0.6,
                row=1, col=1,
            )
            fig.add_vline(
                x=pos, line_width=1, line_dash="dot",
                line_color="#ffd166", opacity=0.6,
                row=2, col=1,
            )
            fig.add_annotation(
                x=pos, y=max(wt_lp.max(), mut_lp.max()) * 0.98,
                text=f"{m['ref']}→{m['alt']}",
                showarrow=True, arrowhead=2, arrowsize=0.8,
                arrowcolor="#ffd166",
                font=dict(size=10, color="#ffd166", family="JetBrains Mono"),
                bgcolor="rgba(13,18,32,0.8)",
                bordercolor="#ffd166",
                borderwidth=1,
                row=1, col=1,
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0d1220",
        font=dict(family="JetBrains Mono, monospace", color="#8892a4", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=50, b=40),
        height=520,
    )

    for anno in fig.layout.annotations:
        if hasattr(anno, "text") and anno.text in [
            "Per-position log P(next base)",
            "ΔLog P  (Mutant − WT)",
        ]:
            anno.font = dict(
                family="JetBrains Mono", size=11, color="#8892a4"
            )

    fig.update_xaxes(
        gridcolor="#1e293b", zeroline=False,
        title_text="Position (bp)", row=2, col=1,
        title_font=dict(size=10),
    )
    fig.update_yaxes(gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title_text="log P", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Δ log P", row=2, col=1, title_font=dict(size=10))

    return fig


def build_composition_chart(wt_seq: str, mut_seq: str) -> go.Figure:
    """Side-by-side nucleotide composition."""
    base_colors = {"A": "#06d6a0", "T": "#ef476f", "C": "#118ab2", "G": "#ffd166"}
    bases = ["A", "C", "G", "T"]

    wt_counts = [wt_seq.upper().count(b) / len(wt_seq) * 100 for b in bases]
    mut_counts = [mut_seq.upper().count(b) / len(mut_seq) * 100 for b in bases]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Wild-type", x=bases, y=wt_counts,
        marker_color=[base_colors[b] for b in bases],
        opacity=0.7,
        text=[f"{v:.1f}%" for v in wt_counts],
        textposition="outside",
        textfont=dict(size=10, color="#8892a4"),
    ))
    fig.add_trace(go.Bar(
        name="Mutant", x=bases, y=mut_counts,
        marker_color=[base_colors[b] for b in bases],
        opacity=0.35,
        marker_line=dict(color=[base_colors[b] for b in bases], width=1.5),
        text=[f"{v:.1f}%" for v in mut_counts],
        textposition="outside",
        textfont=dict(size=10, color="#555e6e"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0d1220",
        barmode="group",
        font=dict(family="JetBrains Mono, monospace", color="#8892a4", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=30, b=40),
        height=280,
        yaxis_title="% Composition",
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def logits_to_base_probs(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits [L, 512] to probabilities over A/C/G/T at each position.

    Returns shape [L, 4] with columns ordered A, C, G, T.
    """
    base_indices = [BASE_TO_ASCII["A"], BASE_TO_ASCII["C"],
                    BASE_TO_ASCII["G"], BASE_TO_ASCII["T"]]
    base_logits = logits[:, base_indices]
    max_vals = np.max(base_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(base_logits - max_vals)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs


def build_entropy_chart(wt_logits: np.ndarray, mut_logits: np.ndarray,
                        mutations: list[dict]) -> go.Figure:
    """Per-position entropy (model uncertainty) for WT and mutant.

    logits[i] predicts position i+1, so we use logits[:-1] aligned to positions 1..L-1.
    """
    def entropy(logits):
        probs = logits_to_base_probs(logits[:-1])
        ent = -np.sum(probs * np.log2(probs + 1e-12), axis=-1)
        return ent

    wt_ent = entropy(wt_logits)
    mut_ent = entropy(mut_logits)
    x = np.arange(1, len(wt_ent) + 1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4],
                        subplot_titles=["Per-position entropy (bits)",
                                        "ΔEntropy (Mutant − WT)"])

    fig.add_trace(go.Scatter(x=x, y=wt_ent, name="WT", line=dict(color="#06d6a0", width=1.2),
                             opacity=0.85), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=mut_ent, name="Mutant", line=dict(color="#ef476f", width=1.2),
                             opacity=0.85), row=1, col=1)

    delta_ent = mut_ent - wt_ent
    colors = ["#ef476f" if d > 0 else "#06d6a0" for d in delta_ent]
    fig.add_trace(go.Bar(x=x, y=delta_ent, marker_color=colors, opacity=0.7,
                         showlegend=False), row=2, col=1)

    for m in mutations:
        for row_i in [1, 2]:
            fig.add_vline(x=m["pos"], line_width=1, line_dash="dot",
                          line_color="#ffd166", opacity=0.6, row=row_i, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1220",
        font=dict(family="JetBrains Mono, monospace", color="#8892a4", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=50, b=40), height=460,
    )
    for anno in fig.layout.annotations:
        anno.font = dict(family="JetBrains Mono", size=11, color="#8892a4")
    fig.update_xaxes(gridcolor="#1e293b", zeroline=False,
                     title_text="Position (bp)", row=2, col=1)
    fig.update_yaxes(gridcolor="#1e293b", zeroline=False,
                     title_text="H (bits)", row=1, col=1)
    fig.update_yaxes(gridcolor="#1e293b", title_text="ΔH", row=2, col=1)
    return fig


def build_acgt_heatmap(logits: np.ndarray, seq: str, label: str,
                       mutations: list[dict]) -> go.Figure:
    """Heatmap of P(A), P(C), P(G), P(T) at each position, with actual base marked.

    logits[i] predicts seq[i+1] (autoregressive), so we align logits[:-1] with seq[1:].
    Position labels correspond to the *predicted* position (1 through L-1).
    """
    probs = logits_to_base_probs(logits[:-1])  # drop last (predicts beyond seq)
    bases = ["A", "C", "G", "T"]
    seq_u = seq.upper()
    target_seq = seq_u[1:]  # logits[i] predicts token i+1
    positions = list(range(1, len(seq_u)))

    fig = go.Figure(data=go.Heatmap(
        z=probs.T, x=positions, y=bases,
        colorscale=[[0, "#0d1220"], [0.25, "#118ab2"], [0.5, "#06d6a0"],
                     [0.75, "#ffd166"], [1.0, "#ef476f"]],
        colorbar=dict(title=dict(text="P", font=dict(size=10))),
        hovertemplate="Pos %{x}<br>Base %{y}<br>P = %{z:.3f}<extra></extra>",
    ))

    actual_y = []
    actual_x = []
    for i, ch in enumerate(target_seq):
        if ch in bases:
            actual_x.append(positions[i])
            actual_y.append(ch)

    fig.add_trace(go.Scatter(
        x=actual_x, y=actual_y, mode="markers",
        marker=dict(symbol="square-open", size=6, color="white", line=dict(width=1.5)),
        name="Actual base", hoverinfo="skip",
    ))

    for m in mutations:
        if m["pos"] in positions:
            fig.add_vline(x=m["pos"], line_width=1, line_dash="dot",
                          line_color="#ffd166", opacity=0.8)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1220",
        font=dict(family="JetBrains Mono, monospace", color="#8892a4", size=11),
        title=dict(text=f"{label} — P(next base) at each position",
                   font=dict(size=12, color="#8892a4")),
        margin=dict(l=50, r=30, t=50, b=40), height=220,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        xaxis_title="Position (bp)",
    )
    fig.update_xaxes(gridcolor="#1e293b")
    return fig


def build_disagreement_chart(logits: np.ndarray, seq: str) -> tuple[go.Figure, int, float]:
    """Show positions where the model's top prediction differs from the actual base.

    logits[i] predicts seq[i+1], so we align logits[:-1] with seq[1:].
    """
    probs = logits_to_base_probs(logits[:-1])
    bases = ["A", "C", "G", "T"]
    base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_u = seq.upper()
    target_seq = seq_u[1:]
    n = len(target_seq)
    positions = list(range(1, len(seq_u)))

    predicted = [bases[np.argmax(probs[i])] for i in range(n)]
    actual_prob = [probs[i, base_idx[target_seq[i]]] for i in range(n)]
    disagree = [1 if predicted[i] != target_seq[i] else 0 for i in range(n)]

    n_disagree = sum(disagree)
    agreement_pct = (1 - n_disagree / n) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=positions, y=actual_prob, name="P(actual base)",
        marker_color=["rgba(6,214,160,0.6)" if not disagree[i] else "rgba(239,71,111,0.6)"
                       for i in range(n)],
        hovertemplate="Pos %{x}<br>Actual: " + "%{customdata[0]}" +
                      "<br>Predicted: %{customdata[1]}<br>P(actual)=%{y:.3f}<extra></extra>",
        customdata=[(target_seq[i], predicted[i]) for i in range(n)],
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1220",
        font=dict(family="JetBrains Mono, monospace", color="#8892a4", size=11),
        margin=dict(l=60, r=30, t=40, b=40), height=280,
        xaxis_title="Position (bp)", yaxis_title="P(actual base)",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig, n_disagree, agreement_pct


# ──────────────────────────────────────────────
# Run analysis
# ──────────────────────────────────────────────

run_col1, run_col2, run_col3 = st.columns([1, 1, 1])
with run_col2:
    run_btn = st.button("Analyze Fitness", use_container_width=True)

if run_btn:
    wt_seq = wt_input.strip().replace("\n", "").replace(" ", "").upper()
    mut_seq = mut_input.strip().replace("\n", "").replace(" ", "").upper()

    if not wt_seq or not mut_seq:
        st.error("Please provide both wild-type and mutant sequences.")
        st.stop()

    invalid = set(wt_seq + mut_seq) - {"A", "C", "G", "T"}
    if invalid:
        st.error(f"Invalid characters: {invalid}. Only A/C/G/T allowed.")
        st.stop()

    if len(wt_seq) != len(mut_seq):
        st.warning(
            f"Sequence lengths differ (WT: {len(wt_seq)}, Mut: {len(mut_seq)}). "
            "Scoring will proceed but strand comparison may be limited."
        )

    mutations = find_mutations(wt_seq, mut_seq)
    if not mutations:
        st.info("The two sequences are identical — no mutations detected.")

    st.markdown("---")

    # API calls
    progress = st.progress(0, text="Initializing…")
    t0 = time.time()

    try:
        progress.progress(10, text="Scoring wild-type sequence…")
        wt_layers = forward(wt_seq)
        wt_logits = extract_logits(wt_layers)
        wt_tokens = tokenize(wt_seq)
        wt_score = score_sequence(wt_logits, wt_tokens, method=scoring_method)
        wt_lp = per_position_scores(wt_logits, wt_tokens)

        progress.progress(50, text="Scoring mutant sequence…")
        mut_layers = forward(mut_seq)
        mut_logits = extract_logits(mut_layers)
        mut_tokens = tokenize(mut_seq)
        mut_score = score_sequence(mut_logits, mut_tokens, method=scoring_method)
        mut_lp = per_position_scores(mut_logits, mut_tokens)

        elapsed = time.time() - t0
        progress.progress(100, text=f"Done in {elapsed:.1f}s")
        time.sleep(0.5)
        progress.empty()

    except Exception as e:
        progress.empty()
        st.error(f"API error: {e}")
        st.info(
            "If you see a 404 or 'not found' error, the forward endpoint may not be "
            "available on the free hosted API. Check NVIDIA docs for self-hosted NIM."
        )
        st.stop()

    delta = mut_score - wt_score

    # ── Metrics row ──
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
        'color:#8892a4; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;">'
        "Results</p>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("WT Score", f"{wt_score:.6f}")
    m2.metric("Mutant Score", f"{mut_score:.6f}")
    m3.metric("Δ Score", f"{delta:+.6f}", delta=f"{delta:+.6f}")
    m4.metric("Mutations", str(len(mutations)))
    m5.metric("Seq Length", f"{len(wt_seq)} bp")

    verdict_color = "#ef476f" if delta < -0.001 else "#06d6a0" if delta > 0.001 else "#ffd166"
    if delta < -0.001:
        verdict = "DESTABILIZING"
        verdict_desc = "The mutation likely reduces fitness."
    elif delta > 0.001:
        verdict = "STABILIZING"
        verdict_desc = "The mutation may increase fitness."
    else:
        verdict = "NEUTRAL"
        verdict_desc = "Minimal fitness impact detected."

    st.markdown(
        f'<div style="background:linear-gradient(135deg, {verdict_color}11, {verdict_color}08); '
        f'border:1px solid {verdict_color}33; border-radius:10px; padding:1rem 1.4rem; '
        f'margin:1rem 0 1.5rem 0;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.75rem; '
        f'font-weight:700; color:{verdict_color}; letter-spacing:0.1em;">{verdict}</span>'
        f'<span style="color:#8892a4; font-size:0.85rem; margin-left:1rem;">{verdict_desc}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──
    tab_strand, tab_logprob, tab_entropy, tab_xray, tab_gen, tab_comp = st.tabs(
        ["STRAND VIEW", "LOG-PROBABILITY", "UNCERTAINTY MAP",
         "MODEL X-RAY", "GENERATE", "COMPOSITION"]
    )

    with tab_strand:
        st.markdown(
            render_strand_html(wt_seq, mut_seq, mutations, window_display),
            unsafe_allow_html=True,
        )

        if len(mutations) > 1 or (mutations and mutations[0]["pos"] > window_display):
            st.markdown(
                '<p style="font-size:0.72rem; color:#555e6e; margin-top:0.6rem;">'
                "Showing the first mutation. Adjust the window size in the sidebar, "
                "or scroll the sequence to see more.</p>",
                unsafe_allow_html=True,
            )

        with st.expander("Reverse complement"):
            rc_wt = reverse_complement(wt_seq)
            rc_mut = reverse_complement(mut_seq)
            rc_mutations = find_mutations(rc_wt, rc_mut)
            st.markdown(
                render_strand_html(rc_wt, rc_mut, rc_mutations, window_display),
                unsafe_allow_html=True,
            )

    with tab_logprob:
        fig = build_logprob_chart(wt_lp, mut_lp, mutations)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Score details"):
            st.markdown(
                f"- **WT mean log P**: `{wt_score:.8f}`\n"
                f"- **Mutant mean log P**: `{mut_score:.8f}`\n"
                f"- **Δ (Mut − WT)**: `{delta:+.8f}`\n"
                f"- **Scoring method**: `{scoring_method}`\n"
                f"- **Model**: Evo 2 40B (NVIDIA NIM hosted API)",
            )

    # ── Uncertainty Map (entropy) ──
    with tab_entropy:
        st.markdown(
            '<p style="color:#8892a4; font-size:0.85rem; margin-bottom:1rem;">'
            "Entropy measures the model's <b>uncertainty</b> at each position. "
            "Low entropy = model is confident (position is conserved). "
            "High entropy = model is uncertain (position is variable / tolerant to mutation). "
            "Mutation at a low-entropy site is more likely to be damaging.</p>",
            unsafe_allow_html=True,
        )
        entropy_fig = build_entropy_chart(wt_logits, mut_logits, mutations)
        st.plotly_chart(entropy_fig, use_container_width=True)

        wt_probs = logits_to_base_probs(wt_logits[:-1])
        wt_ent = -np.sum(wt_probs * np.log2(wt_probs + 1e-12), axis=-1)
        for m in mutations:
            pos = m["pos"]
            idx = pos - 1  # wt_ent is indexed from 0 but represents positions 1..L-1
            if 0 <= idx < len(wt_ent):
                ent_val = wt_ent[idx]
                severity = "HIGH IMPACT" if ent_val < 0.5 else "MODERATE" if ent_val < 1.0 else "LOW IMPACT"
                sev_color = "#ef476f" if ent_val < 0.5 else "#ffd166" if ent_val < 1.0 else "#06d6a0"
                st.markdown(
                    f'<div style="background:#0d1220; border:1px solid #1e293b; border-radius:8px; '
                    f'padding:0.8rem 1.2rem; margin:0.4rem 0; font-family:\'JetBrains Mono\',monospace; font-size:0.78rem;">'
                    f'Mutation <span style="color:#ffd166;">{m["ref"]}→{m["alt"]}</span> at '
                    f'position <span style="color:#ffd166;">{pos}</span>: '
                    f'WT entropy = <span style="color:{sev_color};">{ent_val:.3f} bits</span> '
                    f'→ <span style="color:{sev_color}; font-weight:700;">{severity}</span> '
                    f'<span style="color:#555e6e;">(max possible = 2.0 bits)</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Model X-Ray (ACGT heatmap + disagreement) ──
    with tab_xray:
        st.markdown(
            '<p style="color:#8892a4; font-size:0.85rem; margin-bottom:1rem;">'
            "Peer inside the model's predictions. The heatmap shows the probability "
            "Evo 2 assigns to each base (A/C/G/T) at every position. White squares mark "
            "the actual base. The bar chart below shows where the model <b>disagrees</b> "
            "with the actual sequence — low bars mean the model thinks that base is unlikely.</p>",
            unsafe_allow_html=True,
        )

        wt_hm = build_acgt_heatmap(wt_logits, wt_seq, "Wild-type", mutations)
        st.plotly_chart(wt_hm, use_container_width=True)

        mut_hm = build_acgt_heatmap(mut_logits, mut_seq, "Mutant", mutations)
        st.plotly_chart(mut_hm, use_container_width=True)

        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
            'color:#8892a4; text-transform:uppercase; letter-spacing:0.1em; '
            'margin:1.5rem 0 0.6rem 0;">Model Agreement</p>',
            unsafe_allow_html=True,
        )

        d1, d2 = st.columns(2)
        with d1:
            wt_dis_fig, wt_n_dis, wt_agree = build_disagreement_chart(wt_logits, wt_seq)
            st.markdown(
                f'<p style="font-size:0.78rem; color:#06d6a0; font-family:\'JetBrains Mono\',monospace;">'
                f'WT: {wt_agree:.1f}% agreement ({wt_n_dis} disagreements)</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(wt_dis_fig, use_container_width=True)
        with d2:
            mut_dis_fig, mut_n_dis, mut_agree = build_disagreement_chart(mut_logits, mut_seq)
            st.markdown(
                f'<p style="font-size:0.78rem; color:#ef476f; font-family:\'JetBrains Mono\',monospace;">'
                f'Mutant: {mut_agree:.1f}% agreement ({mut_n_dis} disagreements)</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(mut_dis_fig, use_container_width=True)

        if mutations:
            st.markdown(
                '<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
                'color:#8892a4; text-transform:uppercase; letter-spacing:0.1em; '
                'margin:1.5rem 0 0.6rem 0;">Mutation Site Probability Shift</p>',
                unsafe_allow_html=True,
            )
            wt_probs_full = logits_to_base_probs(wt_logits[:-1])
            mut_probs_full = logits_to_base_probs(mut_logits[:-1])
            bases_list = ["A", "C", "G", "T"]
            base_colors_list = ["#06d6a0", "#118ab2", "#ffd166", "#ef476f"]

            for m in mutations:
                pos = m["pos"]
                idx = pos - 1
                if idx < 0 or idx >= len(wt_probs_full):
                    continue
                fig_shift = go.Figure()
                fig_shift.add_trace(go.Bar(
                    x=bases_list, y=wt_probs_full[idx], name="WT",
                    marker_color=base_colors_list, opacity=0.7,
                    text=[f"{v:.3f}" for v in wt_probs_full[idx]],
                    textposition="outside", textfont=dict(size=9, color="#8892a4"),
                ))
                fig_shift.add_trace(go.Bar(
                    x=bases_list, y=mut_probs_full[idx], name="Mutant",
                    marker_color=base_colors_list, opacity=0.35,
                    marker_line=dict(color=base_colors_list, width=1.5),
                    text=[f"{v:.3f}" for v in mut_probs_full[idx]],
                    textposition="outside", textfont=dict(size=9, color="#555e6e"),
                ))
                fig_shift.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0e17", plot_bgcolor="#0d1220",
                    barmode="group",
                    font=dict(family="JetBrains Mono", color="#8892a4", size=11),
                    title=dict(text=f"Position {pos}: {m['ref']} → {m['alt']}",
                               font=dict(size=11, color="#ffd166")),
                    margin=dict(l=50, r=30, t=40, b=30), height=250,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="center", x=0.5),
                    yaxis_title="Probability",
                )
                fig_shift.update_xaxes(gridcolor="#1e293b")
                fig_shift.update_yaxes(gridcolor="#1e293b")
                st.plotly_chart(fig_shift, use_container_width=True)

    # ── Generate (DNA completion) ──
    with tab_gen:
        st.markdown(
            '<p style="color:#8892a4; font-size:0.85rem; margin-bottom:1rem;">'
            "Use Evo 2 as a <b>DNA autocomplete engine</b>. Give it the wild-type sequence "
            "as a prompt and see what it generates next. High sampled probabilities "
            "mean the model is confident about the continuation.</p>",
            unsafe_allow_html=True,
        )

        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            n_tokens = st.slider("Tokens to generate", 10, 200, 50, 10)
        with gen_col2:
            gen_temp = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1,
                                 help="Lower = more deterministic, higher = more creative")

        gen_top_k = st.select_slider("Top-k sampling", options=[1, 2, 3, 4], value=3)

        if st.button("Generate continuation", use_container_width=True):
            with st.spinner(f"Generating {n_tokens} tokens from WT prompt…"):
                try:
                    gen_result = generate(wt_seq, num_tokens=n_tokens,
                                          temperature=gen_temp, top_k=gen_top_k)
                    gen_seq = gen_result.get("sequence", "")
                    gen_probs = gen_result.get("sampled_probs", [])

                    st.markdown(
                        '<p style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
                        'color:#8892a4; text-transform:uppercase; letter-spacing:0.1em; '
                        'margin:1rem 0 0.5rem 0;">Generated Sequence</p>',
                        unsafe_allow_html=True,
                    )

                    base_colors = {"A": "#06d6a0", "T": "#ef476f",
                                   "C": "#118ab2", "G": "#ffd166"}
                    colored_chars = "".join(
                        f'<span style="color:{base_colors.get(ch.upper(), "#8892a4")};">{ch}</span>'
                        for ch in gen_seq
                    )
                    st.markdown(
                        f'<div style="background:#0d1220; border:1px solid #1e293b; '
                        f'border-radius:10px; padding:1rem 1.2rem; '
                        f'font-family:\'JetBrains Mono\',monospace; font-size:0.78rem; '
                        f'line-height:1.8; word-break:break-all;">{colored_chars}</div>',
                        unsafe_allow_html=True,
                    )

                    if gen_probs:
                        avg_conf = np.mean(gen_probs)
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Tokens generated", len(gen_seq))
                        mc2.metric("Avg confidence", f"{avg_conf:.3f}")
                        mc3.metric("GC%", f"{(gen_seq.upper().count('G') + gen_seq.upper().count('C')) / max(len(gen_seq), 1) * 100:.1f}%")

                        prob_fig = go.Figure()
                        prob_colors = ["#06d6a0" if p > 0.7 else "#ffd166" if p > 0.4
                                       else "#ef476f" for p in gen_probs]
                        prob_fig.add_trace(go.Bar(
                            x=list(range(len(gen_probs))), y=gen_probs,
                            marker_color=prob_colors, opacity=0.8,
                            hovertemplate="Token %{x}<br>P = %{y:.3f}<extra></extra>",
                        ))
                        prob_fig.update_layout(
                            template="plotly_dark", paper_bgcolor="#0a0e17",
                            plot_bgcolor="#0d1220",
                            font=dict(family="JetBrains Mono", color="#8892a4", size=11),
                            margin=dict(l=50, r=30, t=30, b=40), height=250,
                            xaxis_title="Generated token index",
                            yaxis_title="Sampled probability",
                        )
                        prob_fig.update_xaxes(gridcolor="#1e293b")
                        prob_fig.update_yaxes(gridcolor="#1e293b", range=[0, 1])
                        st.plotly_chart(prob_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Generation error: {e}")

    with tab_comp:
        comp_fig = build_composition_chart(wt_seq, mut_seq)
        st.plotly_chart(comp_fig, use_container_width=True)

        gc_wt = (wt_seq.count("G") + wt_seq.count("C")) / len(wt_seq) * 100
        gc_mut = (mut_seq.count("G") + mut_seq.count("C")) / len(mut_seq) * 100
        c1, c2 = st.columns(2)
        c1.metric("WT GC%", f"{gc_wt:.1f}%")
        c2.metric("Mutant GC%", f"{gc_mut:.1f}%")

    # ── Raw data download ──
    st.markdown("---")
    import json

    results_json = json.dumps(
        {
            "wt_score": wt_score,
            "mut_score": mut_score,
            "delta": delta,
            "verdict": verdict,
            "scoring_method": scoring_method,
            "wt_length": len(wt_seq),
            "mut_length": len(mut_seq),
            "mutations": mutations,
            "wt_per_position_logprobs": wt_lp.tolist(),
            "mut_per_position_logprobs": mut_lp.tolist(),
        },
        indent=2,
    )
    st.download_button(
        "Download results (JSON)",
        data=results_json,
        file_name="evo2_fitness_results.json",
        mime="application/json",
    )
