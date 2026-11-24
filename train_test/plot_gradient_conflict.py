from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "ce": "#D55E00",
    "supcon": "#0072B2",
    "accent1": "#009E73",
    "accent2": "#CC79A7",
    "accent3": "#E69F00",
    "accent4": "#56B4E9",
    "neutral": "#4D4D4D",
}

LINE_STYLES = {
    "ce": {"color": PALETTE["ce"], "linestyle": "-", "marker": "o"},
    "supcon": {"color": PALETTE["supcon"], "linestyle": "--", "marker": "s"},
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "savefig.dpi": 300,
        }
    )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _load_log(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        p = p / "gradient_conflict_log.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _ordered_layers(logs: Dict[str, Dict[str, Any]]) -> List[str]:
    names = set()
    for payload in logs.values():
        names.update(payload.get("summary", {}).keys())
    return sorted(names, key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 999)


def _sort_exit_keys(keys: List[str]) -> List[str]:
    return sorted(
        set(keys),
        key=lambda s: (0, int(s)) if str(s).lstrip("-").isdigit() else (1, str(s)),
    )


def _pick_primary_layer(layers: List[str]) -> str:
    if "layer_1" in layers:
        return "layer_1"
    return layers[0]


def _plot_fig1_cosine_heatmaps(logs: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    layers = _ordered_layers(logs)
    if not layers:
        return
    exp_keys = list(logs.keys())
    selected_layers = layers[: min(3, len(layers))]
    nrows = len(exp_keys)
    ncols = len(selected_layers)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols + 0.8, 3.4 * nrows + 0.6),
        squeeze=False,
        layout="constrained",
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = mpl.colormaps["RdBu_r"]
    last_im = None

    for r, ek in enumerate(exp_keys):
        title_prefix = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon-ST"
        summary = logs[ek].get("summary", {})
        for c, ln in enumerate(selected_layers):
            ax = axes[r, c]
            info = summary.get(ln)
            if not info or not info.get("pairwise_cosine_matrix"):
                ax.set_visible(False)
                continue
            mat = np.array(info["pairwise_cosine_matrix"], dtype=np.float64)
            cr_mat = np.array(info["pairwise_conflict_rate_matrix"], dtype=np.float64)
            exits = [f"L{x}" for x in info["active_exits"]]
            last_im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
            ax.set_xticks(range(len(exits)))
            ax.set_yticks(range(len(exits)))
            ax.set_xticklabels(exits)
            ax.set_yticklabels(exits)
            ax.grid(False)
            for i in range(len(exits)):
                for j in range(len(exits)):
                    val = mat[i, j]
                    cr_pct = cr_mat[i, j] * 100.0
                    txt = f"{val:.2f}\n({cr_pct:.0f}%)" if i != j else "1.00"
                    txt_color = "white" if abs(val) > 0.55 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=txt_color)
            ax.set_title(
                f"{title_prefix} | {ln}\nMean cos={info['mean_pairwise_cosine']:.3f}, CR={info['conflict_rate']*100:.1f}%"
            )

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.85, label="Mean Pairwise Cosine Similarity $c_{ij}$")
    fig.suptitle("Pairwise Gradient Cosine Similarity & Conflict Rate (% steps with $c_{ij} < 0$)")
    _save_fig(fig, out_dir, "fig1_cosine_similarity_heatmaps")


def _plot_fig2_conflict_trajectories(logs: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    layers = _ordered_layers(logs)
    if not layers:
        return
    primary_layer = _pick_primary_layer(layers)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout="constrained")
    panels = [
        ("mean_pairwise_cosine", "Mean Pairwise Cosine $\\bar{c}(t)$", 0.0),
        ("conflict_rate", "Pairwise Conflict Rate $\\mathrm{CR}(t)$", None),
        ("cancellation_ratio", "Gradient Cancellation Ratio $\\rho_{\\mathrm{cancel}}(t)$", None),
        ("condition_number", "Spectral Condition Number $\\log_{10} \\kappa(G)$", None),
    ]

    for ax, (metric_key, ylabel, hline) in zip(axes.flat, panels):
        for ek, payload in logs.items():
            label = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon-ST"
            style = LINE_STYLES[ek]
            steps: List[int] = []
            vals: List[float] = []
            for rec in payload.get("step_logs", []):
                if primary_layer in rec["layers"]:
                    v = float(rec["layers"][primary_layer][metric_key])
                    if metric_key == "condition_number":
                        v = float(np.log10(max(1.0, v)))
                    steps.append(int(rec["step"]))
                    vals.append(v)
            if steps:
                ax.plot(
                    steps,
                    vals,
                    label=label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=4,
                    linewidth=1.8,
                )
        if hline is not None:
            ax.axhline(hline, color=PALETTE["neutral"], linestyle=":", linewidth=1.1)
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} at {primary_layer}")
        ax.legend(frameon=False)

    fig.suptitle(f"Gradient Conflict Evolution Across Training ({primary_layer})")
    _save_fig(fig, out_dir, "fig2_conflict_trajectories")


def _plot_fig3_descent_alignment_hsr(logs: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    layers = _ordered_layers(logs)
    if not layers:
        return
    primary_layer = _pick_primary_layer(layers)
    raw_exits: List[str] = []
    for payload in logs.values():
        info = payload.get("summary", {}).get(primary_layer)
        if info and info.get("active_exits"):
            raw_exits.extend(info["active_exits"])
    exits = _sort_exit_keys(raw_exits)
    if not exits:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    x = np.arange(len(exits))
    width = 0.36
    exp_keys = list(logs.keys())

    for idx, ek in enumerate(exp_keys):
        label = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon-ST"
        info = logs[ek].get("summary", {}).get(primary_layer, {})
        align_vals = [float(info.get("mean_descent_alignment", {}).get(ex, 0.0)) for ex in exits]
        hsr_vals = [float(info.get("harmful_step_ratio", {}).get(ex, 0.0)) * 100.0 for ex in exits]
        offset = (idx - (len(exp_keys) - 1) / 2.0) * width
        color = PALETTE[ek]
        hatch = None if ek == "ce" else "//"

        axes[0].bar(x + offset, align_vals, width=width, label=label, color=color, hatch=hatch, edgecolor="black", linewidth=0.6)
        axes[1].bar(x + offset, hsr_vals, width=width, label=label, color=color, hatch=hatch, edgecolor="black", linewidth=0.6)

    axes[0].axhline(0.0, color=PALETTE["neutral"], linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"L{e}" for e in exits])
    axes[0].set_ylabel("Mean Descent Alignment $a_k$")
    axes[0].set_title(f"Effective Descent Alignment at {primary_layer}")
    axes[0].legend(frameon=False)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"L{e}" for e in exits])
    axes[1].set_ylabel("Harmful Step Ratio HSR (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_title(f"Harmful Steps ($a_k < 0$) at {primary_layer}")
    axes[1].legend(frameon=False)

    fig.suptitle("Per-Exit Optimization Harm Analysis (MGDA / CAGrad / ConFIG)")
    _save_fig(fig, out_dir, "fig3_descent_alignment_hsr")


def _plot_fig4_norm_dominance(logs: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    layers = _ordered_layers(logs)
    if not layers:
        return
    primary_layer = _pick_primary_layer(layers)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    exp_keys = list(logs.keys())

    raw_exits: List[str] = []
    for payload in logs.values():
        info = payload.get("summary", {}).get(primary_layer)
        if info and info.get("active_exits"):
            raw_exits.extend(info["active_exits"])
    exits = _sort_exit_keys(raw_exits)

    if exits:
        x = np.arange(len(exits))
        width = 0.36
        for idx, ek in enumerate(exp_keys):
            label = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon-ST"
            info = logs[ek].get("summary", {}).get(primary_layer, {})
            norms = [float(info.get("mean_weighted_norms", {}).get(ex, 0.0)) for ex in exits]
            offset = (idx - (len(exp_keys) - 1) / 2.0) * width
            hatch = None if ek == "ce" else "//"
            axes[0].bar(x + offset, norms, width=width, label=label, color=PALETTE[ek], hatch=hatch, edgecolor="black", linewidth=0.6)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"L{e}" for e in exits])
        axes[0].set_ylabel("Mean Weighted Gradient Norm $\\|w_k g_k\\|_2$")
        axes[0].set_title(f"Per-Exit Gradient Magnitude at {primary_layer}")
        axes[0].legend(frameon=False)

    for ek in exp_keys:
        label = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon-ST"
        style = LINE_STYLES[ek]
        steps: List[int] = []
        doms: List[float] = []
        for rec in logs[ek].get("step_logs", []):
            if primary_layer in rec["layers"]:
                steps.append(int(rec["step"]))
                doms.append(float(rec["layers"][primary_layer]["dominance_ratio"]))
        if steps:
            axes[1].plot(
                steps,
                doms,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4,
                linewidth=1.8,
            )
    axes[1].set_xlabel("Optimization Step")
    axes[1].set_ylabel("Dominance Ratio $D(t) = \\max \\|w_k g_k\\| / \\min \\|w_k g_k\\|$")
    axes[1].set_title(f"Gradient Norm Dominance Trajectory ({primary_layer})")
    axes[1].legend(frameon=False)

    fig.suptitle("Gradient Scale Imbalance Across Exits (Gradient Equilibrium / Nash-MTL)")
    _save_fig(fig, out_dir, "fig4_norm_dominance")


def _plot_fig5_loss_landscapes(logs: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    valid_keys = [ek for ek in logs if logs[ek].get("loss_landscape")]
    if not valid_keys:
        return

    fig, axes = plt.subplots(
        len(valid_keys),
        3,
        figsize=(12.2, 3.8 * len(valid_keys) + 0.6),
        squeeze=False,
        layout="constrained",
    )
    surface_specs = [
        ("shallow_loss", "shallow_min_coord", "Shallow Exit"),
        ("deep_loss", "deep_min_coord", "Deep Exit"),
        ("total_loss", "total_min_coord", "Combined Multi-Exit"),
    ]

    for r, ek in enumerate(valid_keys):
        ls = logs[ek]["loss_landscape"]
        alphas = np.array(ls["alphas"], dtype=np.float64)
        betas = np.array(ls["betas"], dtype=np.float64)
        aa, bb = np.meshgrid(alphas, betas)
        prefix = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon-ST"
        dist = float(ls.get("min_distance", 0.0))

        for c, (grid_key, coord_key, surf_label) in enumerate(surface_specs):
            ax = axes[r, c]
            zz = np.array(ls[grid_key], dtype=np.float64)
            cs = ax.contourf(aa, bb, zz, levels=15, cmap="viridis")
            ax.contour(aa, bb, zz, levels=10, colors="white", linewidths=0.5, alpha=0.6)
            fig.colorbar(cs, ax=ax, shrink=0.85)
            ax.scatter([0.0], [0.0], color="#E69F00", marker="*", s=110, edgecolor="black", zorder=5, label="$\\theta^*$")
            mc = ls[coord_key]
            ax.scatter([mc[0]], [mc[1]], color="#D55E00", marker="X", s=75, edgecolor="white", zorder=6, label="Local min")
            ax.set_xlabel("$\\alpha$ (Shallow grad dir)")
            ax.set_ylabel("$\\beta$ (Orthogonal deep dir)")
            extra = f" | $\\Delta_{{\\min}}={dist:.2f}$" if c == 2 else ""
            exit_tag = (
                f"L{ls['shallow_exit']}"
                if c == 0
                else (f"L{ls['deep_exit']}" if c == 1 else "Total")
            )
            ax.set_title(f"{prefix}\n{surf_label} ({exit_tag}){extra}", fontsize=9.8)
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    fig.suptitle("Filter-Normalized 2D Multi-Exit Loss Landscapes Around $\\theta^*$ (Li et al., NeurIPS 2018)")
    _save_fig(fig, out_dir, "fig5_loss_landscapes_2d")


def _plot_fig6_exit_performance(logs: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")

    for ek, payload in logs.items():
        label = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon-ST (Stage 1)"
        style = LINE_STYLES[ek]
        ep_logs = payload.get("epoch_logs", [])
        if ep_logs:
            epochs = [int(e["epoch"]) for e in ep_logs]
            losses = [float(e["train_total_loss"]) for e in ep_logs]
            axes[0].plot(
                epochs,
                losses,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.8,
            )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Total Loss")
    axes[0].set_title("Training Loss Convergence")
    axes[0].legend(frameon=False)

    exit_keys_sorted: List[str] = []
    acc_by_exp: Dict[str, Dict[str, float]] = {}
    for ek, payload in logs.items():
        if ek == "ce" and payload.get("epoch_logs"):
            acc_by_exp[ek] = payload["epoch_logs"][-1].get("val_accuracy", {})
        elif ek == "supcon" and payload.get("stage2_logs"):
            acc_by_exp[ek] = payload["stage2_logs"][-1].get("val_accuracy", {})
        if acc_by_exp.get(ek):
            exit_keys_sorted = sorted(acc_by_exp[ek].keys(), key=lambda s: int(s))

    if exit_keys_sorted and acc_by_exp:
        x = np.arange(len(exit_keys_sorted))
        width = 0.36
        keys_present = list(acc_by_exp.keys())
        for idx, ek in enumerate(keys_present):
            label = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon + Linear Probe"
            vals = [float(acc_by_exp[ek].get(ex, 0.0)) for ex in exit_keys_sorted]
            offset = (idx - (len(keys_present) - 1) / 2.0) * width
            hatch = None if ek == "ce" else "//"
            axes[1].bar(x + offset, vals, width=width, label=label, color=PALETTE[ek], hatch=hatch, edgecolor="black", linewidth=0.6)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"L{ex}" for ex in exit_keys_sorted])
        axes[1].set_ylim(0.0, 105.0)
        axes[1].set_ylabel("Validation Top-1 Accuracy (%)")
        axes[1].set_title("Per-Exit Validation Accuracy")
        axes[1].legend(frameon=False)

    fig.suptitle("Optimization Convergence & Downstream Multi-Exit Accuracy")
    _save_fig(fig, out_dir, "fig6_exit_performance")


def _write_summary_tables(logs: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ek, payload in logs.items():
        exp_name = "Exp 1: Hamed CE" if ek == "ce" else "Exp 2: SupCon-ST"
        ls_dist = float(payload.get("loss_landscape", {}).get("min_distance", 0.0))
        vram = float(payload.get("peak_vram_mb", 0.0))
        runtime = float(payload.get("runtime_sec", 0.0))
        weighting = str(payload.get("config", {}).get("weighting_mode", ""))

        final_accs: Dict[str, float] = {}
        if ek == "ce" and payload.get("epoch_logs"):
            final_accs = payload["epoch_logs"][-1].get("val_accuracy", {})
        elif ek == "supcon" and payload.get("stage2_logs"):
            final_accs = payload["stage2_logs"][-1].get("val_accuracy", {})

        final_exit_acc = 0.0
        if final_accs:
            last_key = max(
                final_accs.keys(),
                key=lambda s: int(s) if str(s).lstrip("-").isdigit() else -1,
            )
            final_exit_acc = float(final_accs[last_key])

        summary_items = sorted(
            payload.get("summary", {}).items(),
            key=lambda kv: int(kv[0].split("_")[-1]) if kv[0].split("_")[-1].isdigit() else 999,
        )
        for layer_name, info in summary_items:
            hsr_vals = list(info.get("harmful_step_ratio", {}).values())
            max_hsr = float(max(hsr_vals)) if hsr_vals else 0.0
            mean_hsr = float(np.mean(hsr_vals)) if hsr_vals else 0.0
            rows.append(
                {
                    "experiment": exp_name,
                    "weighting_mode": weighting,
                    "probed_layer": layer_name,
                    "mean_pairwise_cosine": round(float(info["mean_pairwise_cosine"]), 4),
                    "conflict_rate_pct": round(float(info["conflict_rate"]) * 100.0, 2),
                    "mean_hsr_pct": round(mean_hsr * 100.0, 2),
                    "max_hsr_pct": round(max_hsr * 100.0, 2),
                    "mean_cancellation_ratio": round(float(info["mean_cancellation_ratio"]), 4),
                    "mean_dominance_ratio": round(float(info["mean_dominance_ratio"]), 3),
                    "median_condition_number": round(float(info["median_condition_number"]), 2),
                    "landscape_min_dist": round(ls_dist, 4),
                    "final_exit_acc_pct": round(final_exit_acc, 2),
                    "peak_vram_mb": round(vram, 1),
                    "runtime_sec": round(runtime, 2),
                }
            )

    csv_path = out_dir / "summary_table.csv"
    md_path = out_dir / "summary_table.md"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        header = "| " + " | ".join(fieldnames) + " |"
        sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
        lines = [header, sep]
        for r in rows:
            lines.append("| " + " | ".join(str(r[k]) for k in fieldnames) + " |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        csv_path.write_text("", encoding="utf-8")
        md_path.write_text("No summary rows available.\n", encoding="utf-8")

    return rows


def generate_all_visualizations(
    exp1_log_path: Optional[Union[str, Path]] = None,
    exp2_log_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "outputs/figures",
) -> Dict[str, Any]:
    _configure_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs: Dict[str, Dict[str, Any]] = {}
    log1 = _load_log(exp1_log_path)
    log2 = _load_log(exp2_log_path)
    if log1 is not None:
        logs["ce"] = log1
    if log2 is not None:
        logs["supcon"] = log2

    if not logs:
        raise FileNotFoundError(
            f"No experiment logs found at exp1={exp1_log_path} or exp2={exp2_log_path}"
        )

    _plot_fig1_cosine_heatmaps(logs, out_dir)
    _plot_fig2_conflict_trajectories(logs, out_dir)
    _plot_fig3_descent_alignment_hsr(logs, out_dir)
    _plot_fig4_norm_dominance(logs, out_dir)
    _plot_fig5_loss_landscapes(logs, out_dir)
    _plot_fig6_exit_performance(logs, out_dir)
    rows = _write_summary_tables(logs, out_dir)
    return {"output_dir": str(out_dir), "num_experiments": len(logs), "summary_rows": rows}
