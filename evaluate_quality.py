#!/usr/bin/env python3
"""
Blind Quality Evaluation for Semantic Conditioning Test

After running the main experiment, use this to score code quality
WITHOUT knowing which condition produced each sample.

Quality dimensions (1-5 each):
1. Edge Cases: Does it handle unexpected inputs without being prompted?
2. Error Handling: Defensive coding, clear error messages
3. Idiomaticity: Pythonic style, good naming, clean structure
4. Documentation: Comments beyond the minimum, clear intent
5. Efficiency: Reasonable algorithmic choices (not premature optimization)

After scoring, run reveal_results.py to see correlation with conditions.
"""

import json
import signal
import sys
from pathlib import Path

REQUIRED_KEYS = {"blind_id", "problem_id", "prompt", "completion"}


RUBRIC = """
QUALITY EVALUATION RUBRIC
=========================

For each code sample, rate these dimensions from 1-5:

1. EDGE CASES (unprompted handling)
   1 = No edge case handling
   2 = Handles one obvious edge case  
   3 = Handles several common edge cases
   4 = Handles edge cases thoughtfully (empty inputs, type issues)
   5 = Comprehensive edge case handling that anticipates problems

2. ERROR HANDLING
   1 = No error handling, will crash on bad input
   2 = Basic try/except, generic handling
   3 = Appropriate exceptions, some error messages
   4 = Good defensive coding, clear error messages
   5 = Robust error handling with helpful feedback

3. IDIOMATICITY (Pythonic style)
   1 = Non-idiomatic, could be translated from C
   2 = Functional but clunky
   3 = Reasonable Python style
   4 = Clean, Pythonic, good naming
   5 = Elegant, would pass senior code review

4. DOCUMENTATION (beyond minimum)
   1 = No comments or documentation
   2 = Minimal docstring only
   3 = Some inline comments
   4 = Clear docstring + meaningful comments
   5 = Excellent documentation explaining why, not just what

5. GUT CHECK: "Would I ship this?"
   1 = No, significant issues
   2 = With reservations, needs work
   3 = Acceptable, typical code
   4 = Good, minor improvements possible
   5 = Yes, this is quality work

Note: Score based on the CODE ONLY, not whether it passes tests.
      A correct but ugly solution should score lower than an elegant one.
"""


def load_blind_data(filepath: str) -> list[dict]:
    """Load and validate the blind evaluation data."""
    with open(filepath) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of samples, got {type(data).__name__}")

    for i, item in enumerate(data):
        missing = REQUIRED_KEYS - item.keys()
        if missing:
            raise ValueError(f"Sample {i} missing required keys: {missing}")

    return data


def evaluate_interactively(data: list[dict], output_file: str):
    """Run interactive evaluation session."""

    # Load existing scores if any
    scores = {}
    if Path(output_file).exists():
        with open(output_file) as f:
            scores = json.load(f)

    def save_and_exit(signum=None, frame=None):
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"\nProgress saved to {output_file}")
        print(f"Scored {len(scores)} / {len(data)} samples")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)

    print(RUBRIC)
    print("\n" + "="*60)
    print(f"Evaluating {len(data)} code samples")
    print("Enter scores as: edge,error,idiom,doc,ship (e.g., 3,2,4,3,3)")
    print("Enter 's' to skip, 'q' to quit and save progress, Ctrl+C to save and exit")
    print("="*60 + "\n")

    if scores:
        print(f"Loaded {len(scores)} existing scores\n")
    
    for i, item in enumerate(data):
        blind_id = item["blind_id"]
        
        if blind_id in scores:
            print(f"[{i+1}/{len(data)}] {blind_id} - already scored, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(data)}] Sample: {blind_id}")
        print(f"Problem: {item['problem_id']}")
        # NOTE: 'passed' intentionally hidden to prevent bias during blind evaluation
        print("-"*60)
        print("PROMPT:")
        print(item["prompt"])
        print("-"*60)
        print("COMPLETION:")
        print(item["completion"])
        print("="*60)
        
        while True:
            response = input("\nScores (edge,error,idiom,doc,ship) or s/q: ").strip().lower()
            
            if response == 'q':
                save_and_exit()
            
            if response == 's':
                print("Skipped")
                break
            
            try:
                parts = [int(x.strip()) for x in response.split(',')]
                if len(parts) != 5:
                    print("Need exactly 5 scores")
                    continue
                if not all(1 <= x <= 5 for x in parts):
                    print("Scores must be 1-5")
                    continue
                
                scores[blind_id] = {
                    "edge_cases": parts[0],
                    "error_handling": parts[1],
                    "idiomaticity": parts[2],
                    "documentation": parts[3],
                    "ship_it": parts[4],
                    "total": sum(parts),
                }
                print(f"Recorded: {parts} (total: {sum(parts)})")
                break
                
            except ValueError:
                print("Invalid format. Use: 3,2,4,3,3")
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"\nEvaluation complete! Saved to {output_file}")
    return scores


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_quality.py <blind_eval_TIMESTAMP.json>")
        print("\nThis will create a scores file for later analysis.")
        sys.exit(1)
    
    blind_file = sys.argv[1]
    data = load_blind_data(blind_file)

    # Derive output filename safely
    if "blind_eval_" in blind_file:
        output_file = blind_file.replace("blind_eval_", "scores_")
    else:
        # Fallback: append _scores before extension
        p = Path(blind_file)
        output_file = str(p.with_stem(p.stem + "_scores"))

    if Path(output_file).resolve() == Path(blind_file).resolve():
        print(f"Error: output file would overwrite input file")
        sys.exit(1)
    
    scores = evaluate_interactively(data, output_file)
    
    if scores:
        print(f"\nScored {len(scores)} samples")
        avg_total = sum(s["total"] for s in scores.values()) / len(scores)
        print(f"Average total score: {avg_total:.2f} / 25")


if __name__ == "__main__":
    main()
