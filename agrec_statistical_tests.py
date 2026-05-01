"""
AGRec Statistical Testing Framework
====================================
This script provides:
1. Multi-seed experiment runner (template)
2. Paired t-test & Wilcoxon signed-rank test
3. Effect size (Cohen's d)
4. Results formatting with mean ± std
5. Unified comparison table generation (LaTeX)
6. α distribution analysis per user group

Usage:
    python agrec_statistical_tests.py

Adjust the `run_single_experiment()` function to call your actual training pipeline.
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
import json
import os
from collections import defaultdict

# ============================================================
# SECTION 1: Multi-Seed Experiment Runner (Template)
# ============================================================

# TODO: Replace this with your actual training + evaluation pipeline
def run_single_experiment(model_name: str, dataset: str, seed: int) -> dict:
    """
    Template: runs one experiment and returns metrics.
    
    Replace the body with your actual training code, e.g.:
        set_seed(seed)
        model = build_model(model_name, config)
        trainer = Trainer(model, dataset, seed=seed)
        trainer.train()
        metrics = trainer.evaluate()
        return metrics
    
    Expected return format:
    {
        "recall@5":  float,
        "recall@10": float,
        "recall@20": float,
        "ndcg@5":    float,
        "ndcg@10":   float,
        "ndcg@20":   float,
        "mrr@10":    float,
    }
    """
    raise NotImplementedError(
        "Connect this to your actual training pipeline. "
        "Return a dict of metric_name -> float."
    )


def run_all_experiments(
    models: list,
    datasets: list,
    seeds: list = [42, 123, 456, 789, 1024],
    save_path: str = "experiment_results.json",
):
    """
    Run every (model, dataset, seed) combination and save results.
    """
    results = {}  # key: (model, dataset, seed) -> metrics dict

    total = len(models) * len(datasets) * len(seeds)
    done = 0

    for model_name in models:
        for dataset in datasets:
            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] Running {model_name} on {dataset} (seed={seed})...")
                try:
                    metrics = run_single_experiment(model_name, dataset, seed)
                    key = f"{model_name}__{dataset}__{seed}"
                    results[key] = metrics
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

    # Save to disk
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {save_path}")
    return results


# ============================================================
# SECTION 2: Load & Organize Results
# ============================================================

def load_results(path: str = "experiment_results.json") -> dict:
    """Load saved experiment results."""
    with open(path) as f:
        return json.load(f)


def organize_results(raw_results: dict) -> pd.DataFrame:
    """
    Convert raw results dict into a tidy DataFrame.
    Columns: model, dataset, seed, recall@5, recall@10, ..., mrr@10
    """
    rows = []
    for key, metrics in raw_results.items():
        model, dataset, seed = key.split("__")
        row = {"model": model, "dataset": dataset, "seed": int(seed)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# SECTION 3: Statistical Tests
# ============================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def pairwise_significance_test(
    df: pd.DataFrame,
    target_model: str = "AGRec",
    metric: str = "recall@10",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each dataset, run paired t-test and Wilcoxon signed-rank test
    comparing target_model vs every other model.
    
    Returns a DataFrame with columns:
        dataset, baseline, metric, 
        target_mean, target_std, baseline_mean, baseline_std,
        improvement%, t_stat, t_pvalue, wilcoxon_stat, wilcoxon_pvalue,
        cohens_d, significant_t, significant_w
    """
    datasets = df["dataset"].unique()
    baselines = [m for m in df["model"].unique() if m != target_model]
    
    rows = []
    for dataset in datasets:
        df_d = df[df["dataset"] == dataset]
        target_scores = df_d[df_d["model"] == target_model][metric].values
        
        if len(target_scores) < 2:
            print(f"WARNING: {target_model} on {dataset} has < 2 runs, skipping.")
            continue
        
        for baseline in baselines:
            base_scores = df_d[df_d["model"] == baseline][metric].values
            
            if len(base_scores) < 2:
                print(f"WARNING: {baseline} on {dataset} has < 2 runs, skipping.")
                continue
            
            # Ensure paired (same seeds in same order)
            n = min(len(target_scores), len(base_scores))
            t_scores = target_scores[:n]
            b_scores = base_scores[:n]
            
            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(t_scores, b_scores)
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            # Requires n >= 6 ideally; with 5 seeds it's marginal but acceptable
            try:
                w_stat, w_pval = stats.wilcoxon(t_scores, b_scores, alternative='two-sided')
            except ValueError:
                # All differences are zero
                w_stat, w_pval = 0.0, 1.0
            
            # Effect size
            d = cohens_d(t_scores, b_scores)
            
            # Improvement
            base_mean = b_scores.mean()
            improvement = ((t_scores.mean() - base_mean) / base_mean * 100) if base_mean != 0 else float('inf')
            
            rows.append({
                "dataset": dataset,
                "baseline": baseline,
                "metric": metric,
                "target_mean": t_scores.mean(),
                "target_std": t_scores.std(ddof=1),
                "baseline_mean": base_mean,
                "baseline_std": b_scores.std(ddof=1),
                "improvement%": improvement,
                "t_stat": t_stat,
                "t_pvalue": t_pval,
                "wilcoxon_stat": w_stat,
                "wilcoxon_pvalue": w_pval,
                "cohens_d": d,
                "significant_t": t_pval < alpha,
                "significant_w": w_pval < alpha,
            })
    
    return pd.DataFrame(rows)


def full_significance_report(
    df: pd.DataFrame,
    target_model: str = "AGRec",
    metrics: list = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run significance tests across all metrics."""
    if metrics is None:
        metrics = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@10"]
    
    all_results = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        result = pairwise_significance_test(df, target_model, metric, alpha)
        all_results.append(result)
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


# ============================================================
# SECTION 4: LaTeX Table Generation
# ============================================================

def generate_unified_latex_table(
    df: pd.DataFrame,
    metrics: list = None,
    target_model: str = "AGRec",
    sig_df: pd.DataFrame = None,
) -> str:
    """
    Generate a unified LaTeX comparison table with mean ± std.
    Marks statistically significant improvements with † (t-test) or ‡ (Wilcoxon).
    Best results are bolded.
    
    Output format matches IEEE Access style.
    """
    if metrics is None:
        metrics = ["recall@10", "ndcg@10", "mrr@10"]
    
    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique(), key=lambda m: m != target_model)  # target last
    
    # Build significance lookup: (dataset, baseline, metric) -> significant?
    sig_lookup = {}
    if sig_df is not None:
        for _, row in sig_df.iterrows():
            key = (row["dataset"], row["baseline"], row["metric"])
            sig_lookup[key] = row.get("significant_t", False)
    
    # Header
    n_metrics = len(metrics)
    metric_labels = {
        "recall@5": "R@5", "recall@10": "R@10", "recall@20": "R@20",
        "ndcg@5": "N@5", "ndcg@10": "N@10", "ndcg@20": "N@20",
        "mrr@10": "MRR@10",
    }
    
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Overall performance comparison. "
                  r"Best results are \textbf{bold}. "
                  r"$\dagger$ indicates statistically significant improvement "
                  r"over the baseline (paired $t$-test, $p < 0.05$).}")
    lines.append(r"\label{tab:main_results}")
    
    col_spec = "ll" + "c" * (n_metrics * len(datasets))
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Dataset header row
    header1 = r" & "
    for ds in datasets:
        header1 += r" & \multicolumn{" + str(n_metrics) + r"}{c}{\textbf{" + ds.replace("_", r"\_") + r"}}"
    header1 += r" \\"
    lines.append(header1)
    
    # Metric header row
    header2 = r"\textbf{Model} "
    for ds in datasets:
        for m in metrics:
            header2 += r" & " + metric_labels.get(m, m)
    header2 += r" \\"
    lines.append(r"\cmidrule(lr){1-1}" + "".join(
        [r"\cmidrule(lr){" + str(2 + i * n_metrics) + "-" + str(1 + (i + 1) * n_metrics) + "}"
         for i in range(len(datasets))]
    ))
    lines.append(header2)
    lines.append(r"\midrule")
    
    # Compute best values per (dataset, metric)
    best_vals = {}
    for ds in datasets:
        for m in metrics:
            vals = df[df["dataset"] == ds].groupby("model")[m].mean()
            best_vals[(ds, m)] = vals.max()
    
    # Model rows
    for model in models:
        row = r"\textbf{" + model.replace("_", r"\_") + r"}" if model == target_model else model.replace("_", r"\_")
        
        for ds in datasets:
            df_sub = df[(df["dataset"] == ds) & (df["model"] == model)]
            for m in metrics:
                if len(df_sub) == 0:
                    row += r" & -"
                    continue
                
                mean_val = df_sub[m].mean()
                std_val = df_sub[m].std(ddof=1) if len(df_sub) > 1 else 0.0
                
                cell = f"{mean_val:.4f}"
                if std_val > 0:
                    cell += f"$\\pm${std_val:.4f}"
                
                # Bold if best
                if abs(mean_val - best_vals.get((ds, m), -1)) < 1e-6:
                    cell = r"\textbf{" + cell + "}"
                
                # Significance marker
                if model == target_model and sig_lookup.get((ds, model, m), False):
                    cell += r"$^\dagger$"
                
                row += " & " + cell
        
        row += r" \\"
        lines.append(row)
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


# ============================================================
# SECTION 5: α Distribution Analysis
# ============================================================

def analyze_alpha_distribution(
    alpha_values: np.ndarray,
    user_history_lengths: np.ndarray,
    cold_threshold: int = 5,
    active_threshold: int = 20,
    save_path: str = "alpha_analysis.json",
) -> dict:
    """
    Analyze the distribution of the router's α values across user groups.
    
    Args:
        alpha_values: array of shape (n_users,), the learned fusion weight α for each user
        user_history_lengths: array of shape (n_users,), number of interactions per user
        cold_threshold: users with ≤ this many interactions are "cold-start"
        active_threshold: users with > this many interactions are "active"
    
    Returns:
        dict with per-group statistics and KS-test results
    """
    cold_mask = user_history_lengths <= cold_threshold
    moderate_mask = (user_history_lengths > cold_threshold) & (user_history_lengths <= active_threshold)
    active_mask = user_history_lengths > active_threshold
    
    groups = {
        f"cold_start (≤{cold_threshold})": alpha_values[cold_mask],
        f"moderate ({cold_threshold+1}-{active_threshold})": alpha_values[moderate_mask],
        f"active (>{active_threshold})": alpha_values[active_mask],
    }
    
    analysis = {}
    for name, vals in groups.items():
        if len(vals) == 0:
            continue
        analysis[name] = {
            "count": int(len(vals)),
            "mean_alpha": float(vals.mean()),
            "std_alpha": float(vals.std()),
            "median_alpha": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
        }
    
    # KS-test between cold-start and active
    group_names = list(groups.keys())
    if len(groups) >= 2:
        g1 = groups[group_names[0]]
        g2 = groups[group_names[-1]]
        if len(g1) > 0 and len(g2) > 0:
            ks_stat, ks_pval = stats.ks_2samp(g1, g2)
            analysis["ks_test_cold_vs_active"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_pval),
                "significant": bool(ks_pval < 0.05),
            }
    
    # Mann-Whitney U test between cold and active
    if len(groups) >= 2:
        g1 = groups[group_names[0]]
        g2 = groups[group_names[-1]]
        if len(g1) > 0 and len(g2) > 0:
            u_stat, u_pval = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            analysis["mannwhitney_cold_vs_active"] = {
                "statistic": float(u_stat),
                "p_value": float(u_pval),
                "significant": bool(u_pval < 0.05),
            }
    
    with open(save_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Alpha analysis saved to {save_path}")
    
    return analysis


def generate_alpha_plot_code() -> str:
    """
    Returns matplotlib code to create the α distribution visualization.
    Copy this into your plotting script.
    """
    return '''
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (10, 4),
})

def plot_alpha_distribution(alpha_values, user_history_lengths,
                            cold_threshold=5, active_threshold=20,
                            dataset_name="Dataset"):
    """
    Plot violin + box plot of α distribution per user group.
    """
    cold_mask   = user_history_lengths <= cold_threshold
    moderate_mask = (user_history_lengths > cold_threshold) & (user_history_lengths <= active_threshold)
    active_mask = user_history_lengths > active_threshold

    data = [
        alpha_values[cold_mask],
        alpha_values[moderate_mask],
        alpha_values[active_mask],
    ]
    labels = [
        f"Cold-start\\n(≤{cold_threshold} interactions)\\nn={cold_mask.sum()}",
        f"Moderate\\n({cold_threshold+1}-{active_threshold})\\nn={moderate_mask.sum()}",
        f"Active\\n(>{active_threshold})\\nn={active_mask.sum()}",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Violin plot
    parts = axes[0].violinplot(data, positions=[1, 2, 3], showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.6)
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Fusion Weight α")
    axes[0].set_title(f"α Distribution by User Group ({dataset_name})")
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="α=0.5")
    axes[0].legend()

    # Right: Box plot with individual points
    bp = axes[1].boxplot(data, positions=[1, 2, 3], widths=0.5, patch_artist=True)
    colors = ["#DD8452", "#55A868", "#4C72B0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, d in enumerate(data):
        jitter = np.random.normal(0, 0.04, size=len(d))
        axes[1].scatter(np.full_like(d, i + 1) + jitter, d, alpha=0.15, s=8, color="black")
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Fusion Weight α")
    axes[1].set_title(f"α Box Plot ({dataset_name})")

    plt.tight_layout()
    plt.savefig(f"alpha_distribution_{dataset_name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved alpha_distribution_{dataset_name}.pdf")
'''


# ============================================================
# SECTION 6: Per-User-Group Performance Breakdown
# ============================================================

def per_group_performance(
    model_predictions: dict,
    ground_truth: dict,
    user_history_lengths: dict,
    cold_threshold: int = 5,
    active_threshold: int = 20,
    k_list: list = None,
) -> pd.DataFrame:
    """
    Compute Recall@K and NDCG@K per user group.
    
    Args:
        model_predictions: {user_id: [ranked list of item_ids]}
        ground_truth:      {user_id: ground_truth_item_id}
        user_history_lengths: {user_id: int}
        k_list: list of K values
    
    Returns:
        DataFrame with columns: group, metric, value
    """
    if k_list is None:
        k_list = [5, 10, 20]
    
    def recall_at_k(pred_list, truth, k):
        return 1.0 if truth in pred_list[:k] else 0.0
    
    def ndcg_at_k(pred_list, truth, k):
        for i, item in enumerate(pred_list[:k]):
            if item == truth:
                return 1.0 / np.log2(i + 2)
        return 0.0
    
    groups = {"cold_start": [], "moderate": [], "active": []}
    
    for uid in model_predictions:
        hist_len = user_history_lengths.get(uid, 0)
        if hist_len <= cold_threshold:
            groups["cold_start"].append(uid)
        elif hist_len <= active_threshold:
            groups["moderate"].append(uid)
        else:
            groups["active"].append(uid)
    
    rows = []
    for group_name, user_ids in groups.items():
        for k in k_list:
            recalls, ndcgs = [], []
            for uid in user_ids:
                pred = model_predictions[uid]
                truth = ground_truth[uid]
                recalls.append(recall_at_k(pred, truth, k))
                ndcgs.append(ndcg_at_k(pred, truth, k))
            
            rows.append({
                "group": group_name,
                "n_users": len(user_ids),
                f"recall@{k}": np.mean(recalls) if recalls else 0,
                f"ndcg@{k}": np.mean(ndcgs) if ndcgs else 0,
            })
    
    # Merge rows by group
    merged = {}
    for row in rows:
        g = row["group"]
        if g not in merged:
            merged[g] = {"group": g, "n_users": row["n_users"]}
        merged[g].update({k: v for k, v in row.items() if k not in ("group", "n_users")})
    
    return pd.DataFrame(list(merged.values()))


# ============================================================
# SECTION 7: Hyperparameter Sensitivity (Template)
# ============================================================

def hyperparameter_sensitivity_template():
    """
    Template for hyperparameter sensitivity experiments.
    Prints the experimental plan.
    """
    plan = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           Hyperparameter Sensitivity Experiment Plan         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Fix all other params at default. Vary one at a time:        ║
    ║                                                              ║
    ║  1. Temporal decay λ:                                        ║
    ║     Values: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]               ║
    ║     Default: 0.1                                             ║
    ║                                                              ║
    ║  2. Entropy regularization λ_ent:                            ║
    ║     Values: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]           ║
    ║     Default: 0.01                                            ║
    ║                                                              ║
    ║  3. LoRA rank r:                                             ║
    ║     Values: [2, 4, 8, 16, 32]                                ║
    ║     Default: 8                                               ║
    ║                                                              ║
    ║  4. Orthogonality loss weight λ_orth:                        ║
    ║     Values: [0.0, 0.01, 0.05, 0.1, 0.5]                    ║
    ║     Default: (set your default)                              ║
    ║                                                              ║
    ║  5. Negative sampling ratio (N_hard:N_med:N_easy):           ║
    ║     Values: [5:5:10, 10:5:5, 10:10:0, 15:5:0, 20:0:0]     ║
    ║     Default: 10:5:5                                          ║
    ║                                                              ║
    ║  Report: Recall@10 and NDCG@10 on all 4 datasets            ║
    ║  Plot: Line chart (x=param value, y=metric)                  ║
    ║  Seeds: Use 3 seeds minimum, report mean ± std               ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(plan)
    
    # Code template for running one sensitivity sweep
    code = '''
# Example: sweep over λ (temporal decay)
lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
seeds = [42, 123, 456]
dataset = "Baby_Products"  # or loop over all datasets

results = []
for lam in lambda_values:
    for seed in seeds:
        config = default_config.copy()
        config["temporal_decay"] = lam
        config["seed"] = seed
        
        metrics = run_single_experiment("AGRec", dataset, seed, extra_config=config)
        results.append({
            "lambda": lam, "seed": seed,
            "recall@10": metrics["recall@10"],
            "ndcg@10": metrics["ndcg@10"],
        })

df = pd.DataFrame(results)
summary = df.groupby("lambda").agg(["mean", "std"])
print(summary)
'''
    return code


# ============================================================
# DEMO: Using synthetic data to show output format
# ============================================================

def demo_with_synthetic_data():
    """
    Demonstrates the full pipeline using synthetic experiment results.
    Replace with your real data.
    """
    print("=" * 70)
    print("DEMO: Statistical Testing Framework (Synthetic Data)")
    print("=" * 70)
    
    np.random.seed(42)
    
    models = ["AGRec", "AlphaRec", "CoLLM", "LLM-SRec", "RecGPT",
              "Fine-Only", "Coarse-Only", "Fixed-Fusion",
              "SASRec", "Bert4Rec", "GRU4Rec"]
    datasets = ["Baby_Products", "Musical_Instruments", "Steam", "Yelp"]
    seeds = [42, 123, 456, 789, 1024]
    
    # Synthetic baseline performance (approximate from paper + noise)
    base_performance = {
        ("AGRec", "Baby_Products"):         {"recall@10": 0.0242, "ndcg@10": 0.0116, "mrr@10": 0.0085},
        ("AlphaRec", "Baby_Products"):      {"recall@10": 0.0080, "ndcg@10": 0.0064, "mrr@10": 0.0050},
        ("CoLLM", "Baby_Products"):         {"recall@10": 0.0159, "ndcg@10": 0.0093, "mrr@10": 0.0066},
        ("LLM-SRec", "Baby_Products"):      {"recall@10": 0.0029, "ndcg@10": 0.0026, "mrr@10": 0.0008},
        ("RecGPT", "Baby_Products"):        {"recall@10": 0.0007, "ndcg@10": 0.0007, "mrr@10": 0.0001},
        ("Fine-Only", "Baby_Products"):     {"recall@10": 0.0238, "ndcg@10": 0.0112, "mrr@10": 0.0080},
        ("Coarse-Only", "Baby_Products"):   {"recall@10": 0.0011, "ndcg@10": 0.0005, "mrr@10": 0.0003},
        ("Fixed-Fusion", "Baby_Products"):  {"recall@10": 0.0217, "ndcg@10": 0.0105, "mrr@10": 0.0075},
        ("SASRec", "Baby_Products"):        {"recall@10": 0.0185, "ndcg@10": 0.0090, "mrr@10": 0.0065},
        ("Bert4Rec", "Baby_Products"):      {"recall@10": 0.0178, "ndcg@10": 0.0086, "mrr@10": 0.0062},
        ("GRU4Rec", "Baby_Products"):       {"recall@10": 0.0155, "ndcg@10": 0.0074, "mrr@10": 0.0055},
        
        ("AGRec", "Musical_Instruments"):        {"recall@10": 0.0271, "ndcg@10": 0.0137, "mrr@10": 0.0100},
        ("AlphaRec", "Musical_Instruments"):     {"recall@10": 0.0034, "ndcg@10": 0.0026, "mrr@10": 0.0025},
        ("CoLLM", "Musical_Instruments"):        {"recall@10": 0.0290, "ndcg@10": 0.0169, "mrr@10": 0.0124},
        ("LLM-SRec", "Musical_Instruments"):     {"recall@10": 0.0003, "ndcg@10": 0.0002, "mrr@10": 0.0001},
        ("RecGPT", "Musical_Instruments"):       {"recall@10": 0.0004, "ndcg@10": 0.0002, "mrr@10": 0.0000},
        ("Fine-Only", "Musical_Instruments"):    {"recall@10": 0.0266, "ndcg@10": 0.0134, "mrr@10": 0.0098},
        ("Coarse-Only", "Musical_Instruments"):  {"recall@10": 0.0033, "ndcg@10": 0.0015, "mrr@10": 0.0010},
        ("Fixed-Fusion", "Musical_Instruments"): {"recall@10": 0.0267, "ndcg@10": 0.0134, "mrr@10": 0.0098},
        ("SASRec", "Musical_Instruments"):       {"recall@10": 0.0220, "ndcg@10": 0.0110, "mrr@10": 0.0082},
        ("Bert4Rec", "Musical_Instruments"):     {"recall@10": 0.0210, "ndcg@10": 0.0105, "mrr@10": 0.0078},
        ("GRU4Rec", "Musical_Instruments"):      {"recall@10": 0.0195, "ndcg@10": 0.0095, "mrr@10": 0.0070},
        
        ("AGRec", "Steam"):         {"recall@10": 0.0738, "ndcg@10": 0.0363, "mrr@10": 0.0280},
        ("AlphaRec", "Steam"):      {"recall@10": 0.0193, "ndcg@10": 0.0169, "mrr@10": 0.0152},
        ("CoLLM", "Steam"):         {"recall@10": 0.0613, "ndcg@10": 0.0305, "mrr@10": 0.0125},
        ("LLM-SRec", "Steam"):      {"recall@10": 0.0054, "ndcg@10": 0.0039, "mrr@10": 0.0003},
        ("RecGPT", "Steam"):        {"recall@10": 0.0013, "ndcg@10": 0.0008, "mrr@10": 0.0001},
        ("Fine-Only", "Steam"):     {"recall@10": 0.0724, "ndcg@10": 0.0355, "mrr@10": 0.0270},
        ("Coarse-Only", "Steam"):   {"recall@10": 0.0089, "ndcg@10": 0.0040, "mrr@10": 0.0030},
        ("Fixed-Fusion", "Steam"):  {"recall@10": 0.0732, "ndcg@10": 0.0359, "mrr@10": 0.0275},
        ("SASRec", "Steam"):        {"recall@10": 0.0580, "ndcg@10": 0.0290, "mrr@10": 0.0220},
        ("Bert4Rec", "Steam"):      {"recall@10": 0.0560, "ndcg@10": 0.0275, "mrr@10": 0.0210},
        ("GRU4Rec", "Steam"):       {"recall@10": 0.0510, "ndcg@10": 0.0250, "mrr@10": 0.0190},
        
        ("AGRec", "Yelp"):         {"recall@10": 0.0082, "ndcg@10": 0.0040, "mrr@10": 0.0030},
        ("AlphaRec", "Yelp"):      {"recall@10": 0.0210, "ndcg@10": 0.0154, "mrr@10": 0.0137},
        ("CoLLM", "Yelp"):         {"recall@10": 0.0162, "ndcg@10": 0.0084, "mrr@10": 0.0061},
        ("LLM-SRec", "Yelp"):      {"recall@10": 0.0001, "ndcg@10": 0.0001, "mrr@10": 0.0013},
        ("RecGPT", "Yelp"):        {"recall@10": 0.0002, "ndcg@10": 0.0002, "mrr@10": 0.0000},
        ("Fine-Only", "Yelp"):     {"recall@10": 0.0047, "ndcg@10": 0.0020, "mrr@10": 0.0015},
        ("Coarse-Only", "Yelp"):   {"recall@10": 0.0036, "ndcg@10": 0.0017, "mrr@10": 0.0012},
        ("Fixed-Fusion", "Yelp"):  {"recall@10": 0.0077, "ndcg@10": 0.0035, "mrr@10": 0.0026},
        ("SASRec", "Yelp"):        {"recall@10": 0.0165, "ndcg@10": 0.0080, "mrr@10": 0.0058},
        ("Bert4Rec", "Yelp"):      {"recall@10": 0.0158, "ndcg@10": 0.0076, "mrr@10": 0.0055},
        ("GRU4Rec", "Yelp"):       {"recall@10": 0.0142, "ndcg@10": 0.0068, "mrr@10": 0.0050},
    }
    
    # Generate synthetic multi-seed results with realistic variance
    raw_results = {}
    for (model, dataset), base_metrics in base_performance.items():
        for seed in seeds:
            noise_scale = 0.03  # 3% relative noise
            metrics = {}
            for metric, val in base_metrics.items():
                noisy_val = val * (1 + np.random.normal(0, noise_scale))
                metrics[metric] = max(0, noisy_val)
            
            key = f"{model}__{dataset}__{seed}"
            raw_results[key] = metrics
    
    # Organize into DataFrame
    df = organize_results(raw_results)
    
    # --- 1. Significance Tests ---
    print("\n" + "=" * 70)
    print("1. STATISTICAL SIGNIFICANCE TESTS (AGRec vs Baselines)")
    print("=" * 70)
    
    sig_df = full_significance_report(df, target_model="AGRec", metrics=["recall@10", "ndcg@10"])
    
    # Pretty print
    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        subset = sig_df[(sig_df["dataset"] == dataset) & (sig_df["metric"] == "recall@10")]
        for _, row in subset.iterrows():
            sig_marker = "✓" if row["significant_t"] else "✗"
            print(f"  vs {row['baseline']:15s}  "
                  f"Δ={row['improvement%']:+7.2f}%  "
                  f"t={row['t_stat']:+6.3f}  p={row['t_pvalue']:.4f}  "
                  f"d={row['cohens_d']:+.3f}  "
                  f"sig={sig_marker}")
    
    # --- 2. Unified LaTeX Table ---
    print("\n" + "=" * 70)
    print("2. UNIFIED LaTeX TABLE")
    print("=" * 70)
    
    latex = generate_unified_latex_table(df, metrics=["recall@10", "ndcg@10", "mrr@10"], sig_df=sig_df)
    print(latex)
    
    # Save LaTeX
    with open("unified_results_table.tex", "w") as f:
        f.write(latex)
    print("\n[Saved to unified_results_table.tex]")
    
    # --- 3. Summary Table (mean ± std) ---
    print("\n" + "=" * 70)
    print("3. SUMMARY: mean ± std PER MODEL (Recall@10)")
    print("=" * 70)
    
    summary = df.groupby(["dataset", "model"])["recall@10"].agg(["mean", "std"])
    summary["formatted"] = summary.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    
    for dataset in datasets:
        print(f"\n{dataset}:")
        ds_summary = summary.loc[dataset].sort_values("mean", ascending=False)
        for model, row in ds_summary.iterrows():
            marker = " ★" if model == "AGRec" else ""
            print(f"  {model:18s}  {row['formatted']}{marker}")
    
    # --- 4. Alpha Distribution Demo ---
    print("\n" + "=" * 70)
    print("4. α DISTRIBUTION ANALYSIS (Synthetic Demo)")
    print("=" * 70)
    
    n_users = 5000
    history_lengths = np.concatenate([
        np.random.randint(1, 6, size=1500),    # cold-start
        np.random.randint(6, 21, size=2000),   # moderate
        np.random.randint(21, 100, size=1500), # active
    ])
    
    # Simulate α: cold-start users → higher α (more coarse), active → lower α
    alpha_values = np.clip(
        0.7 - 0.005 * history_lengths + np.random.normal(0, 0.12, n_users),
        0.05, 0.95
    )
    
    analysis = analyze_alpha_distribution(alpha_values, history_lengths)
    
    for group, stats_dict in analysis.items():
        if isinstance(stats_dict, dict) and "mean_alpha" in stats_dict:
            print(f"  {group}: mean α = {stats_dict['mean_alpha']:.4f} ± {stats_dict['std_alpha']:.4f}  "
                  f"(median={stats_dict['median_alpha']:.4f}, n={stats_dict['count']})")
    
    if "ks_test_cold_vs_active" in analysis:
        ks = analysis["ks_test_cold_vs_active"]
        print(f"\n  KS-test (cold vs active): stat={ks['statistic']:.4f}, p={ks['p_value']:.2e}, "
              f"significant={ks['significant']}")
    
    if "mannwhitney_cold_vs_active" in analysis:
        mw = analysis["mannwhitney_cold_vs_active"]
        print(f"  Mann-Whitney U (cold vs active): stat={mw['statistic']:.1f}, p={mw['p_value']:.2e}, "
              f"significant={mw['significant']}")
    
    print("\n" + "=" * 70)
    print("PLOTTING CODE (copy into your visualization script):")
    print("=" * 70)
    print(generate_alpha_plot_code())
    
    # --- 5. Hyperparameter Plan ---
    print("\n" + "=" * 70)
    print("5. HYPERPARAMETER SENSITIVITY PLAN")
    print("=" * 70)
    hyperparameter_sensitivity_template()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    demo_with_synthetic_data()
