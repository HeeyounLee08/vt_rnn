"""
top_models.py - Rank model configs at three levels of aggregation.

Auto-detects all available (view, activation) combos from results/.
Evaluates at:
  1. Within activation  — fixed activation, compare across views & models
  2. Within view        — fixed view, compare across activations & models
  3. Overall            — everything pooled

Output saved to results/top_models.txt (and printed).
"""
import io
import sys
import pandas as pd
from pathlib import Path

results_dir = Path(__file__).parent / 'results'

METRICS = {
    'bps':           ('Bits per spike',  True),
    'pearson_r':     ('Pearson r',       True),
    'test_nll':      ('Test NLL',        False),
    'mean_rate_err': ('Mean rate error', False),
}
TOP_N = 5
GROUP_COLS = ['model', 'hidden_size', 'tau_lr', 'view', 'activation']


# ── Auto-detect available configs ────────────────────────────────────────────

def find_combos():
    combos = []
    for view_dir in sorted(results_dir.iterdir()):
        if not view_dir.is_dir() or view_dir.name.startswith('.'):
            continue
        for act_dir in sorted(view_dir.iterdir()):
            if act_dir.is_dir() and list(act_dir.rglob('metrics.csv')):
                combos.append((view_dir.name, act_dir.name))
    return combos


def load_all(combos):
    frames = []
    for view, act in combos:
        act_dir = results_dir / view / act
        csvs = list(act_dir.rglob('metrics.csv'))
        if not csvs:
            continue
        df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        df['view'] = view
        df['activation'] = act
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Ranking helpers ───────────────────────────────────────────────────────────

def _section(buf, title, data, col, higher_better):
    buf.write(f"\n{'─'*72}\n  {title}\n{'─'*72}\n")
    ranked = data.sort_values(col, ascending=not higher_better).head(TOP_N)
    for i, row in enumerate(ranked.itertuples(), 1):
        view_tag = f"[{row.view}/{row.activation}]"
        buf.write(
            f"  {i}. {row.model:<22} h={row.hidden_size:<4} τ_lr={row.tau_lr:<5}"
            f"  {col}={getattr(row, col):+.4f}  {view_tag}\n"
        )


def rank_group(buf, df, group_label):
    buf.write(f"\n\n{'█'*72}\n  {group_label}\n{'█'*72}")

    # Overall across cell types
    overall = df.groupby(GROUP_COLS, as_index=False).agg(
        **{col: (col, 'mean') for col in METRICS}
    )
    buf.write(f"\n\n  ── Averaged across all cell types ──")
    for col, (name, hb) in METRICS.items():
        _section(buf, f"Top {TOP_N} — {name} ({'↑' if hb else '↓'})", overall, col, hb)

    # Per cell type
    for ct in sorted(df['cell_type'].unique()):
        sub = df[df['cell_type'] == ct]
        cname = sub['cell_name'].iloc[0]
        grp = sub.groupby(GROUP_COLS, as_index=False).agg(
            **{col: (col, 'mean') for col in METRICS}
        )
        buf.write(f"\n\n  ── Cell type {ct}: {cname} ──")
        for col, (name, hb) in METRICS.items():
            _section(buf, f"Top {TOP_N} — {name} ({'↑' if hb else '↓'})", grp, col, hb)


# ── Main ──────────────────────────────────────────────────────────────────────

combos = find_combos()
if not combos:
    print(f"No results found under {results_dir}")
    sys.exit(0)

print(f"Found configs: {combos}")
df_all = load_all(combos)
if df_all.empty:
    print("No data loaded.")
    sys.exit(0)

buf = io.StringIO()
buf.write("TOP MODELS REPORT\n" + "═" * 72)

# ── Level 1: Within each activation ──────────────────────────────────────────
activations = sorted(df_all['activation'].unique())
for act in activations:
    sub = df_all[df_all['activation'] == act]
    views_present = sorted(sub['view'].unique())
    rank_group(buf, sub,
               f"LEVEL 1 — WITHIN ACTIVATION: {act.upper()}"
               f"  (views: {', '.join(views_present)})")

# ── Level 2: Within each view ─────────────────────────────────────────────────
views = sorted(df_all['view'].unique())
for view in views:
    sub = df_all[df_all['view'] == view]
    acts_present = sorted(sub['activation'].unique())
    rank_group(buf, sub,
               f"LEVEL 2 — WITHIN VIEW: {view.upper()}"
               f"  (activations: {', '.join(acts_present)})")

# ── Level 3: Overall ─────────────────────────────────────────────────────────
rank_group(buf, df_all,
           f"LEVEL 3 — OVERALL  (all views × activations: "
           f"{', '.join(f'{v}/{a}' for v, a in combos)})")

output = buf.getvalue()
print(output)

out_path = results_dir / 'top_models.txt'
out_path.write_text(output)
print(f"\nSaved to {out_path}")
