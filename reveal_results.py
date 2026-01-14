#!/usr/bin/env python3
"""
Reveal Results: Analyze quality scores by condition

Run this AFTER completing blind evaluation to see if semantic conditioning
correlates with quality scores.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics


def load_data(results_file: str, scores_file: str, key_file: str):
    """Load all data files."""
    with open(results_file) as f:
        results = json.load(f)
    
    with open(scores_file) as f:
        scores = json.load(f)
    
    with open(key_file) as f:
        key = json.load(f)
    
    return results, scores, key


def analyze(results: list, scores: dict, key: dict):
    """Analyze quality scores by condition."""
    
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
    
    print("\n" + "="*70)
    print("QUALITY SCORES BY CONDITION (semantic conditioning test)")
    print("="*70)
    
    # Define quality dimensions
    dimensions = ["edge_cases", "error_handling", "idiomaticity", "documentation", "ship_it", "total"]
    
    # Header
    print(f"\n{'Condition':<20}", end="")
    for dim in dimensions:
        print(f"{dim[:8]:>10}", end="")
    print(f"{'n':>6}")
    print("-"*70)
    
    # Summary stats by condition
    condition_order = ["control", "common_english", "greek_arete", "japanese_shokunin", "combined"]
    
    for condition in condition_order:
        if condition not in by_condition:
            continue
        
        scores_list = by_condition[condition]
        n = len(scores_list)
        
        print(f"{condition:<20}", end="")
        for dim in dimensions:
            values = [s[dim] for s in scores_list]
            mean = statistics.mean(values) if values else 0
            print(f"{mean:>10.2f}", end="")
        print(f"{n:>6}")
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON (vs control)")
    print("="*70)
    
    if "control" not in by_condition:
        print("No control condition found!")
        return
    
    control_totals = [s["total"] for s in by_condition["control"]]
    control_mean = statistics.mean(control_totals)
    control_std = statistics.stdev(control_totals) if len(control_totals) > 1 else 0
    
    print(f"\nControl baseline: {control_mean:.2f} ± {control_std:.2f} (n={len(control_totals)})")
    print()
    
    for condition in ["common_english", "greek_arete", "japanese_shokunin", "combined"]:
        if condition not in by_condition:
            continue
        
        totals = [s["total"] for s in by_condition[condition]]
        mean = statistics.mean(totals)
        std = statistics.stdev(totals) if len(totals) > 1 else 0
        diff = mean - control_mean
        pct_diff = (diff / control_mean) * 100 if control_mean else 0
        
        # Effect size (Cohen's d, rough)
        pooled_std = ((control_std**2 + std**2) / 2) ** 0.5 if control_std and std else 1
        effect = diff / pooled_std if pooled_std else 0
        
        direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"{condition:<20} {mean:.2f} ± {std:.2f}  {direction} {abs(diff):+.2f} ({pct_diff:+.1f}%)  d={effect:.2f}")
    
    # Breakdown by dimension
    print("\n" + "="*70)
    print("DIMENSION BREAKDOWN (mean by condition)")
    print("="*70)
    
    for dim in ["edge_cases", "error_handling", "idiomaticity", "documentation", "ship_it"]:
        print(f"\n{dim.upper()}")
        for condition in condition_order:
            if condition not in by_condition:
                continue
            values = [s[dim] for s in by_condition[condition]]
            mean = statistics.mean(values)
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
    
    print(f"""
Highest quality scores: {best_condition} (mean total: {best_mean:.2f})

What this suggests:
- If rare-token conditions (greek_arete, japanese_shokunin, combined) score higher
  than common_english, the semantic conditioning hypothesis has support.
- If common_english ≈ rare tokens, the effect may be about ANY quality instruction.
- If control ≈ everything else, quality instructions may have minimal impact.

Caveats:
- Small sample sizes limit statistical power
- Blind evaluation introduces evaluator noise
- Single evaluator = potential bias (consider multiple evaluators)
- HumanEval problems may not differentiate well on quality dimensions
""")


def main():
    if len(sys.argv) < 4:
        print("Usage: python reveal_results.py <results_*.json> <scores_*.json> <key_*.json>")
        print("\nExample:")
        print("  python reveal_results.py results_20240115.json scores_20240115.json key_20240115.json")
        sys.exit(1)
    
    results, scores, key = load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    analyze(results, scores, key)


if __name__ == "__main__":
    main()
