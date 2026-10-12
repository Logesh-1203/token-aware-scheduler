import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_benchmark_charts(summary_csv_path: str, raw_csv_path: str, output_dir: str = None) -> list:
    """Generate professional benchmark evaluation charts from real benchmark CSVs."""
    summary_path = Path(summary_csv_path)
    raw_path = Path(raw_csv_path)

    if not summary_path.exists() or not raw_path.exists():
        print(f"Error: CSV files not found: {summary_path}, {raw_path}")
        return []

    out_dir = Path(output_dir) if output_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_path)
    raw_df = pd.read_csv(raw_path)

    run_id = summary_path.stem.replace("benchmark_", "").replace("_summary", "")
    created_plots = []

    # Palette
    colors = {
        "fifo": "#e74c3c",
        "token_aware": "#3498db",
        "adaptive": "#2ecc71",
        "adaptive_least_conn": "#f39c12",
        "priority_only": "#9b59b6",
        "sjf": "#1abc9c",
        "adaptive_oracle": "#27ae60",
        "adaptive_pred": "#2980b9",
        "adaptive_no_aging": "#e67e22",
        "adaptive_no_cost": "#8e44ad",
    }

    # 1. Latency & TTFT Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.25

    policies = summary_df["policy"].tolist()
    p50_lats = summary_df["p50_latency_sec"].tolist()
    p95_lats = summary_df["p95_latency_sec"].tolist()
    ttfts = summary_df["mean_ttft_sec"].tolist()

    ax.bar(x - width, p50_lats, width, label="p50 Latency (s)", color="#3498db")
    ax.bar(x, p95_lats, width, label="p95 Latency (s)", color="#e74c3c")
    ax.bar(x + width, ttfts, width, label="Mean TTFT (s)", color="#2ecc71")

    ax.set_ylabel("Seconds")
    ax.set_title(f"Latency & TTFT by Scheduling Policy ({run_id})")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    p1 = out_dir / f"benchmark_{run_id}_latencies.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    created_plots.append(str(p1))

    # 2. Throughput Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tputs = summary_df["throughput_req_per_sec"].tolist()
    bar_colors = [colors.get(p, "#9b59b6") for p in policies]
    bars = ax.bar(policies, tputs, color=bar_colors, width=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} rps", ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel("Throughput (Requests / Second)")
    ax.set_title(f"System Throughput Comparison ({run_id})")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    p2 = out_dir / f"benchmark_{run_id}_throughput.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    created_plots.append(str(p2))

    # 3. Waiting Time by Profile per Policy
    if "profile" in raw_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        profile_wait = raw_df.groupby(["policy", "profile"])["queue_wait"].mean().unstack()
        profile_wait.plot(kind="bar", ax=ax, colormap="viridis", width=0.6)
        ax.set_ylabel("Mean Queue Wait Time (Seconds)")
        ax.set_title(f"Queue Waiting Time by Client Profile ({run_id})")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=0, fontweight="bold")
        plt.tight_layout()

        p3 = out_dir / f"benchmark_{run_id}_profile_waits.png"
        plt.savefig(p3, dpi=300)
        plt.close()
        created_plots.append(str(p3))

    # 4. Latency CDF (Cumulative Distribution Function)
    fig, ax = plt.subplots(figsize=(9, 5))
    for policy in raw_df["policy"].unique():
        p_data = raw_df[raw_df["policy"] == policy]["completion_latency"].dropna().sort_values()
        if len(p_data) > 0:
            cdf = np.arange(1, len(p_data) + 1) / len(p_data)
            ax.plot(p_data, cdf, label=policy, color=colors.get(policy, "#333333"), linewidth=2.5)

    ax.set_xlabel("Completion Latency (Seconds)")
    ax.set_ylabel("Cumulative Probability (CDF)")
    ax.set_title(f"Latency CDF by Scheduling Policy ({run_id})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    p4 = out_dir / f"benchmark_{run_id}_cdf.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    created_plots.append(str(p4))

    # 5. SLO Attainment and Goodput Comparison
    if "slo_attainment_pct" in summary_df.columns:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        x = np.arange(len(summary_df))
        width = 0.35

        slos = summary_df["slo_attainment_pct"].tolist()
        goodputs = summary_df.get("goodput_req_per_sec", summary_df.get("throughput_req_per_sec", pd.Series([0]*len(summary_df)))).tolist()

        b1 = ax1.bar(x - width/2, slos, width, label="SLO Attainment (%)", color="#2ecc71")
        b2 = ax2.bar(x + width/2, goodputs, width, label="Goodput (req/s)", color="#3498db")

        ax1.set_ylabel("SLO Attainment (%)", color="#27ae60", fontweight="bold")
        ax2.set_ylabel("Goodput (req/s)", color="#2980b9", fontweight="bold")
        ax1.set_title(f"SLO Attainment & Goodput by Policy ({run_id})")
        ax1.set_xticks(x)
        ax1.set_xticklabels(policies, fontweight="bold")
        ax1.set_ylim(0, 105)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        plt.tight_layout()

        p5 = out_dir / f"benchmark_{run_id}_slo_goodput.png"
        plt.savefig(p5, dpi=300)
        plt.close()
        created_plots.append(str(p5))

    print(f"Generated {len(created_plots)} benchmark charts in {out_dir}")
    return created_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark comparison charts from CSVs.")
    parser.add_argument("summary_csv", help="Path to benchmark summary CSV")
    parser.add_argument("raw_csv", help="Path to benchmark raw CSV")
    parser.add_argument("--output_dir", default=None, help="Directory to save PNG plots")
    args = parser.parse_args()

    generate_benchmark_charts(args.summary_csv, args.raw_csv, args.output_dir)
