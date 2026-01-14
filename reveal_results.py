#!/usr/bin/env python3
"""
Reveal Results: Analyze quality scores by condition

Run this AFTER completing blind evaluation to see if semantic conditioning
correlates with quality scores.

Includes:
- Summary statistics by condition
- Permutation test for significance (p-value)
- Bootstrap confidence intervals
- Effect size (Cohen's d with proper pooled std)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics
import random


def load_data(results_file: str, scores_file: str, key_file: str):
    """Load and validate all data files."""
    try:
        with open(results_file) as f:
            raw_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Failed to load results file: {e}")

    try:
        with open(scores_file) as f:
            scores = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Failed to load scores file: {e}")

    try:
        with open(key_file) as f:
            key = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Failed to load key file: {e}")

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(raw_results, dict) and "trials" in raw_results:
        metadata = raw_results.get("metadata", {})
        results = raw_results["trials"]
    else:
        metadata = {}
        results = raw_results

    if not isinstance(results, list):
        raise ValueError("Results must be a list of trial objects")
    if not isinstance(scores, dict):
        raise ValueError("Scores must be a dict mapping blind_id to score data")
    if not isinstance(key, dict):
        raise ValueError("Key must be a dict mapping blind_id to condition")

    return results, scores, key, metadata


def permutation_test(group_a: list[float], group_b: list[float],
                     n_permutations: int = 10000) -> float:
    """
    Two-sided permutation test for difference in means.
    Returns p-value: probability of observing this large a difference by chance.
    """
    if not group_a or not group_b:
        return 1.0

    observed_diff = abs(statistics.mean(group_a) - statistics.mean(group_b))
    combined = group_a + group_b
    n_a = len(group_a)

    count_extreme = 0
    for _ in range(n_permutations):
        random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = abs(statistics.mean(perm_a) - statistics.mean(perm_b))
        if perm_diff >= observed_diff:
            count_extreme += 1

    return count_extreme / n_permutations


def bootstrap_ci(data: list[float], confidence: float = 0.95,
                 n_bootstrap: int = 10000) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean.
    Returns (lower, upper) bounds.
    """
    if len(data) < 2:
        mean = statistics.mean(data) if data else 0
        return (mean, mean)

    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        means.append(statistics.mean(sample))

    means.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return (means[lower_idx], means[upper_idx])


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """
    Cohen's d effect size with proper pooled standard deviation.
    Handles unequal sample sizes correctly.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0

    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)
    n_a = len(group_a)
    n_b = len(group_b)

    # Pooled standard deviation (weighted by degrees of freedom)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = pooled_var ** 0.5

    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


def analyze(results: list, scores: dict, key: dict, metadata: dict):
    """Analyze quality scores by condition."""

    # Print metadata if available
    if metadata:
        print("\n" + "="*70)
        print("EXPERIMENT METADATA")
        print("="*70)
        for k, v in metadata.items():
            print(f"  {k}: {v}")

    # Group scores by condition
    by_condition = defaultdict(list)
    by_problem = defaultdict(list)

    for blind_id, score_data in scores.items():
        condition = key.get(blind_id)
        if condition:
            by_condition[condition].append(score_data)

            # Find problem_id from results
            for r in results:
                if r["blind_id"] == blind_id:
                    by_problem[(condition, r["problem_id"])].append(score_data)
                    break

    if not by_condition:
        print("\nNo scores matched to conditions. Check that key file matches scores.")
        return

    print("\n" + "="*70)
    print("QUALITY SCORES BY CONDITION")
    print("="*70)

    # Define quality dimensions
    dimensions = ["edge_cases", "error_handling", "idiomaticity", "documentation", "ship_it", "total"]

    # Header
    print(f"\n{'Condition':<20}", end="")
    for dim in dimensions:
        print(f"{dim[:8]:>10}", end="")
    print(f"{'n':>6}")
    print("-"*70)

    # Dynamic condition order: control first, then alphabetical
    all_conditions = sorted(by_condition.keys())
    condition_order = []
    if "control" in all_conditions:
        condition_order.append("control")
        all_conditions.remove("control")
    condition_order.extend(all_conditions)

    # Summary stats by condition
    for condition in condition_order:
        if condition not in by_condition:
            continue

        scores_list = by_condition[condition]
        n = len(scores_list)

        print(f"{condition:<20}", end="")
        for dim in dimensions:
            values = [s[dim] for s in scores_list if dim in s]
            mean = statistics.mean(values) if values else 0
            print(f"{mean:>10.2f}", end="")
        print(f"{n:>6}")

    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON (vs control)")
    print("="*70)

    if "control" not in by_condition:
        print("\nNo control condition found - cannot compute relative effects.")
        print("Showing absolute statistics only.\n")

        for condition in condition_order:
            totals = [s["total"] for s in by_condition[condition]]
            mean = statistics.mean(totals)
            std = statistics.stdev(totals) if len(totals) > 1 else 0
            ci_low, ci_high = bootstrap_ci(totals)
            print(f"{condition:<20} {mean:.2f} ± {std:.2f}  95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
        return

    control_totals = [s["total"] for s in by_condition["control"]]
    control_mean = statistics.mean(control_totals)
    control_std = statistics.stdev(control_totals) if len(control_totals) > 1 else 0
    control_ci = bootstrap_ci(control_totals)

    print(f"\nControl baseline: {control_mean:.2f} ± {control_std:.2f}")
    print(f"  95% CI: [{control_ci[0]:.2f}, {control_ci[1]:.2f}]")
    print(f"  n = {len(control_totals)}")
    print()

    for condition in condition_order:
        if condition == "control":
            continue
        if condition not in by_condition:
            continue

        totals = [s["total"] for s in by_condition[condition]]
        mean = statistics.mean(totals)
        std = statistics.stdev(totals) if len(totals) > 1 else 0
        diff = mean - control_mean
        pct_diff = (diff / control_mean) * 100 if control_mean else 0

        # Effect size and significance
        d = cohens_d(totals, control_totals)
        p_value = permutation_test(totals, control_totals)
        ci_low, ci_high = bootstrap_ci(totals)

        direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        sig = "*" if p_value < 0.05 else ""
        sig = "**" if p_value < 0.01 else sig
        sig = "***" if p_value < 0.001 else sig

        print(f"{condition:<20} {mean:.2f} ± {std:.2f}  {direction} {diff:+.2f} ({pct_diff:+.1f}%)")
        print(f"  {'':20} d={d:.2f}  p={p_value:.3f}{sig}  95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

    # Significance legend
    print("\n  * p<0.05  ** p<0.01  *** p<0.001 (permutation test, 10k iterations)")

    # Breakdown by dimension
    print("\n" + "="*70)
    print("DIMENSION BREAKDOWN (mean by condition)")
    print("="*70)

    for dim in ["edge_cases", "error_handling", "idiomaticity", "documentation", "ship_it"]:
        print(f"\n{dim.upper()}")
        for condition in condition_order:
            if condition not in by_condition:
                continue
            values = [s[dim] for s in by_condition[condition] if dim in s]
            mean = statistics.mean(values) if values else 0
            print(f"  {condition:<20} {mean:.2f}")

    # Pass rate integration
    print("\n" + "="*70)
    print("PASS RATE BY CONDITION (from results)")
    print("="*70)

    for condition in condition_order:
        condition_results = [r for r in results if r["condition"] == condition]
        passed = sum(1 for r in condition_results if r["passed"])
        total = len(condition_results)
        rate = passed / total if total > 0 else 0
        print(f"  {condition:<20} {passed:3}/{total:3} = {rate:.1%}")

    # Key insight
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    best_condition = max(by_condition.keys(),
                        key=lambda c: statistics.mean([s["total"] for s in by_condition[c]]))
    best_mean = statistics.mean([s["total"] for s in by_condition[best_condition]])

    # Check for significant differences
    sig_conditions = []
    for condition in condition_order:
        if condition == "control":
            continue
        totals = [s["total"] for s in by_condition[condition]]
        p = permutation_test(totals, control_totals)
        if p < 0.05:
            d = cohens_d(totals, control_totals)
            sig_conditions.append((condition, p, d))

    print(f"""
Highest quality scores: {best_condition} (mean total: {best_mean:.2f})
""")

    if sig_conditions:
        print("Statistically significant differences from control:")
        for cond, p, d in sig_conditions:
            effect_size = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
            print(f"  - {cond}: p={p:.3f}, d={d:.2f} ({effect_size} effect)")
    else:
        print("No statistically significant differences from control (p<0.05).")
        print("This could mean:")
        print("  - Sample size too small to detect effects")
        print("  - Effects genuinely absent")
        print("  - High variance in quality scores")

    print("""
Interpretation guide:
- If rare-token conditions significantly outperform control with p<0.05,
  the semantic conditioning hypothesis has support.
- If generic_quality ≈ common_english ≈ rare tokens, any instruction helps.
- If common_english outperforms generic_quality, specificity matters.
- Effect sizes: d<0.2 trivial, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large

Caveats:
- Permutation test assumes exchangeability under null hypothesis
- Bootstrap CIs are approximate for small samples
- Single evaluator introduces systematic bias
- Consider multiple evaluators for robust conclusions
""")


def main():
    if len(sys.argv) < 4:
        print("Usage: python reveal_results.py <results_*.json> <scores_*.json> <key_*.json>")
        print("\nExample:")
        print("  python reveal_results.py results/results_20240115.json results/scores_20240115.json results/key_20240115.json")
        sys.exit(1)

    try:
        results, scores, key, metadata = load_data(sys.argv[1], sys.argv[2], sys.argv[3])
        analyze(results, scores, key, metadata)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
