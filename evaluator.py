"""
evaluator.py — AirDefend-X Benchmark Suite
===========================================
Produces two separate 2x2 confusion matrices:

  1. AI Layer (Isolation Forest) — evaluated only on RSA-verified frames
       True class:  IMPROPER PHYSICS vs LEGITIMATE
       Predicted:   AI flagged vs AI passed

  2. Identity Layer (RSA-2048) — evaluated on all frames
       True class:  FAKE IDENTITY vs LEGITIMATE
       Predicted:   RSA failed vs RSA passed

Both matrices are printed as ASCII tables in the terminal and saved
as a side-by-side PNG (benchmark_results.png).

Attack injection:
  - Physics attacks: unique registered ICAO, valid RSA signature,
    anomalous flight envelope (vel=60, alt=12000, vr=45) — same profile
    used by the app. Only AI can catch these.
  - Identity attacks: unregistered ICAO, forged signature bytes.
    Only RSA can catch these.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split

import logic as _logic_module
from logic import (
    run_middleware,
    register_and_sign_legitimate,
    prepare_dataframe,
    train_behavioral_model,
    make_physics_attack_row,
    sign_message,
    _registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(title: str, rows: list[str], width: int = 52) -> str:
    """Draw a simple ASCII box around lines of text."""
    border = "+" + "-" * width + "+"
    lines  = [border, f"|  {title:<{width-2}}|", border]
    for r in rows:
        lines.append(f"|  {r:<{width-2}}|")
    lines.append(border)
    return "\n".join(lines)


def _print_2x2(title: str, tp: int, fp: int, fn: int, tn: int,
               pos_label: str, neg_label: str):
    """
    Print a 2x2 confusion matrix as an ASCII table and return metrics.

    Matrix layout (standard):
                     Predicted POS    Predicted NEG
    Actual POS            TP               FN
    Actual NEG            FP               TN
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / (tp + fp + fn + tn)

    w = 56
    sep   = "+" + "-"*18 + "+" + "-"*18 + "+" + "-"*18 + "+"
    head  = f"|{'':18}|{'Predicted: ' + pos_label:^18}|{'Predicted: ' + neg_label:^18}|"
    r1    = f"|{'Actual: ' + pos_label:18}|{tp:^18}|{fn:^18}|"
    r2    = f"|{'Actual: ' + neg_label:18}|{fp:^18}|{tn:^18}|"

    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")
    print(sep)
    print(head)
    print(sep)
    print(r1)
    print(sep)
    print(r2)
    print(sep)

    metrics = [
        f"Precision : {precision:.4f}   ({tp} correctly flagged / {tp+fp} total flagged)",
        f"Recall    : {recall:.4f}   ({tp} caught / {tp+fn} total attacks)",
        f"F1 Score  : {f1:.4f}",
        f"Accuracy  : {accuracy:.4f}",
        f"False Pos : {fp}   (legitimate frames incorrectly flagged)",
        f"False Neg : {fn}   (attacks that slipped through)",
    ]
    print(_box("Results", metrics))
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _plot_2x2(ax, matrix: list[list[int]], title: str,
              pos_label: str, neg_label: str):
    """Draw a styled 2x2 heatmap on the given axes."""
    labels = [pos_label, neg_label]
    arr    = np.array(matrix)

    # Colour each cell: TP/TN = good (green tones), FP/FN = bad (red tones)
    colors = np.array([
        ["#2ecc71", "#e74c3c"],   # TP green, FN red
        ["#e74c3c", "#2ecc71"],   # FP red,   TN green
    ])

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                         color=colors[i][j], alpha=0.7))
            cell_label = {(0,0):"TP",(0,1):"FN",(1,0):"FP",(1,1):"TN"}[(i,j)]
            ax.text(j + 0.5, 1 - i + 0.6, str(arr[i][j]),
                    ha='center', va='center', fontsize=28, fontweight='bold',
                    color='white')
            ax.text(j + 0.5, 1 - i + 0.25, cell_label,
                    ha='center', va='center', fontsize=11, color='white',
                    alpha=0.85)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels([f"Predicted\n{pos_label}", f"Predicted\n{neg_label}"],
                       fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([f"Actual\n{neg_label}", f"Actual\n{pos_label}"],
                       fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(csv_path: str = 'data/opensky.csv'):
    print("\n" + "="*58)
    print("   AirDefend-X  ·  Security Evaluation")
    print("="*58)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    try:
        raw = pd.read_csv(csv_path).dropna()
        print(f"[✓] Loaded {len(raw)} rows from {csv_path}")
    except FileNotFoundError:
        print("[!] CSV not found — using synthetic baseline (300 flights)")
        rng = np.random.default_rng(42)
        n   = 300
        raw = pd.DataFrame({
            'icao24':      [f'REAL_{i:04d}' for i in range(n)],
            'velocity':    rng.uniform(100, 300, n),
            'vertrate':    rng.uniform(-5,  5,   n),
            'geoaltitude': rng.uniform(5000,12000,n),
            'lat':         rng.uniform(20,  60,   n),
            'lon':         rng.uniform(-20, 40,   n),
            'track':       rng.uniform(0,   360,  n),
        })

    raw = prepare_dataframe(raw)

    # ── 2. Train / test split ────────────────────────────────────────────────
    train_raw, test_raw = train_test_split(raw, test_size=0.3, random_state=42)
    print(f"[✓] Train: {len(train_raw)}  |  Test: {len(test_raw)}  (all legitimate)")

    # ── 3. Register & sign ───────────────────────────────────────────────────
    train_df = register_and_sign_legitimate(train_raw)
    test_df  = register_and_sign_legitimate(test_raw)
    train_df['ground_truth'] = 'LEGITIMATE'
    test_df['ground_truth']  = 'LEGITIMATE'

    # ── 4. Train AI on legitimate-only training split ────────────────────────
    print("[✓] Training Isolation Forest on legitimate data only ...")
    train_behavioral_model(train_df)
    _logic_module._trained = True

    # ── 5. Build physics attacks (same profile as app) ───────────────────────
    physics_attacks = []
    n_physics = 40
    for i in range(n_physics):
        base = test_df.iloc[i % len(test_df)].copy()
        base['lat'] += np.random.uniform(-1, 1)
        base['lon'] += np.random.uniform(-1, 1)
        base['time'] = float(i)
        attack_icao = f"ATKP{i:04d}"
        atk = make_physics_attack_row(base, attack_icao)
        atk['ground_truth'] = 'IMPROPER PHYSICS'
        physics_attacks.append(atk)

    # ── 6. Build identity attacks ────────────────────────────────────────────
    identity_attacks = []
    n_identity = 40
    ref = test_df.iloc[0].copy()
    for i in range(n_identity):
        atk = ref.copy()
        atk['icao24']       = f"SPOOF{i:04d}"
        atk['velocity']     = np.random.uniform(150, 320)
        atk['lat']         += np.random.uniform(-1, 1)
        atk['lon']         += np.random.uniform(-1, 1)
        atk['time']         = float(i)
        atk['signature']    = b"FORGED_" + str(i).encode()
        atk['ground_truth'] = 'FAKE IDENTITY'
        identity_attacks.append(atk)

    # ── 7. Combine & shuffle ─────────────────────────────────────────────────
    atk_df    = prepare_dataframe(pd.DataFrame(physics_attacks + identity_attacks))
    full_test = pd.concat([test_df, atk_df], ignore_index=True).sample(
        frac=1, random_state=42).reset_index(drop=True)
    print(f"[✓] Test set: {len(test_df)} legit + {n_physics} physics + "
          f"{n_identity} identity = {len(full_test)} total")

    # ── 8. Run pipeline ───────────────────────────────────────────────────────
    print("[✓] Running AirDefend-X middleware pipeline ...")
    results = run_middleware(full_test)
    results['ground_truth'] = full_test['ground_truth'].values

    # ── 9. Compute 2x2 for AI layer ───────────────────────────────────────────
    # Scope: only RSA-verified frames (auth_passed=True)
    # Positive class: IMPROPER PHYSICS
    ai_scope = results[results['auth_passed'] == True].copy()
    ai_true  = ai_scope['ground_truth'] == 'IMPROPER PHYSICS'
    ai_pred  = ai_scope['status']       == 'IMPROPER PHYSICS'

    ai_tp = int(( ai_true &  ai_pred).sum())
    ai_fp = int((~ai_true &  ai_pred).sum())
    ai_fn = int(( ai_true & ~ai_pred).sum())
    ai_tn = int((~ai_true & ~ai_pred).sum())

    # ── 10. Compute 2x2 for RSA layer ─────────────────────────────────────────
    # Scope: all frames
    # Positive class: FAKE IDENTITY
    id_true = results['ground_truth'] == 'FAKE IDENTITY'
    id_pred = results['status']       == 'FAKE IDENTITY'

    id_tp = int(( id_true &  id_pred).sum())
    id_fp = int((~id_true &  id_pred).sum())
    id_fn = int(( id_true & ~id_pred).sum())
    id_tn = int((~id_true & ~id_pred).sum())

    # ── 11. Print ASCII tables ────────────────────────────────────────────────
    print("\n")
    ai_metrics = _print_2x2(
        "AI BEHAVIORAL LAYER — Isolation Forest  (physics detection)",
        ai_tp, ai_fp, ai_fn, ai_tn,
        pos_label="PHYSICS", neg_label="LEGIT"
    )

    id_metrics = _print_2x2(
        "RSA IDENTITY LAYER — RSA-2048  (identity authentication)",
        id_tp, id_fp, id_fn, id_tn,
        pos_label="FAKE ID", neg_label="LEGIT"
    )

    # ── 12. Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  OVERALL SUMMARY")
    print(f"{'='*58}")
    fmt = "  {:<28} {:>10}  {:>10}"
    print(fmt.format("Metric", "AI Layer", "RSA Layer"))
    print("  " + "-"*54)
    for key, label in [("precision","Precision"), ("recall","Recall"),
                       ("f1","F1 Score"), ("accuracy","Accuracy")]:
        print(fmt.format(label,
                         f"{ai_metrics[key]:.4f}",
                         f"{id_metrics[key]:.4f}"))
    print("  " + "-"*54)
    total_attacks  = n_physics + n_identity
    total_caught   = ai_metrics['tp'] + id_metrics['tp']
    total_missed   = ai_metrics['fn'] + id_metrics['fn']
    print(f"  {'Total attacks injected':<28} {total_attacks:>10}")
    print(f"  {'Total caught':<28} {total_caught:>10}")
    print(f"  {'Total missed (false neg)':<28} {total_missed:>10}")
    print(f"{'='*58}\n")

    # ── 13. Plot side-by-side PNG ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white')

    _plot_2x2(ax1,
              [[ai_tp, ai_fn], [ai_fp, ai_tn]],
              "AI Behavioral Layer\n(Isolation Forest — Physics Detection)",
              "PHYSICS", "LEGIT")

    _plot_2x2(ax2,
              [[id_tp, id_fn], [id_fp, id_tn]],
              "RSA Identity Layer\n(RSA-2048 — Identity Authentication)",
              "FAKE ID", "LEGIT")

    # Metric annotations below each matrix
    for ax, m in [(ax1, ai_metrics), (ax2, id_metrics)]:
        ax.text(1.0, -0.22,
                f"Precision {m['precision']:.3f}  |  Recall {m['recall']:.3f}"
                f"  |  F1 {m['f1']:.3f}  |  Accuracy {m['accuracy']:.3f}",
                transform=ax.transAxes, ha='center', fontsize=9.5,
                color='#cccccc')
        ax.title.set_color('white')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('#cccccc')

    fig.suptitle("AirDefend-X — Security Layer Evaluation", fontsize=15,
                 fontweight='bold', color='white', y=1.01)
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print("[✓] Confusion matrices saved → benchmark_results.png\n")

    return results


if __name__ == "__main__":
    run_benchmark()
