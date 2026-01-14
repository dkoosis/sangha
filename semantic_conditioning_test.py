#!/usr/bin/env python3
"""
Semantic Conditioning Test: Does rare philosophical vocabulary improve code quality?

Hypothesis: Rare tokens like "ἀρετή" (Greek: excellence/virtue) and "職人気質" 
(Japanese: craftsman spirit) may activate different attention patterns than 
overused phrases like "high quality", potentially yielding better code.

Design:
- Use HumanEval benchmark problems (objective correctness via unit tests)
- Compare conditions: control, common English, Greek, Japanese, combined
- Measure: pass rate (objective) + blind quality evaluation (subjective)
"""

import json
import os
import random
import hashlib
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# You'll need: pip install datasets anthropic
from datasets import load_dataset
import anthropic


@dataclass
class Condition:
    name: str
    prefix: str  # Added before the prompt
    suffix: str  # Added after the prompt


CONDITIONS = [
    Condition(
        name="control",
        prefix="",
        suffix=""
    ),
    Condition(
        name="common_english",
        prefix="Write high-quality, production-ready code.\n\n",
        suffix=""
    ),
    Condition(
        name="greek_arete",
        prefix="Write code embodying ἀρετή (excellence).\n\n",
        suffix=""
    ),
    Condition(
        name="japanese_shokunin",
        prefix="Write code with 職人気質 (craftsman spirit).\n\n",
        suffix=""
    ),
    Condition(
        name="combined",
        prefix="Write code embodying ἀρετή and 職人気質.\n\n",
        suffix=""
    ),
]


# HumanEval problems good for quality differentiation (not trivially simple)
# These have room for edge case handling, documentation, idiomatic style
SELECTED_PROBLEMS = [
    "HumanEval/2",   # truncate_number - extract decimal part
    "HumanEval/4",   # mean_absolute_deviation 
    "HumanEval/11",  # string_xor
    "HumanEval/22",  # filter_integers from mixed list
    "HumanEval/29",  # filter_by_prefix
]


@dataclass
class TrialResult:
    problem_id: str
    condition: str
    completion: str
    passed: bool
    error: Optional[str]
    timestamp: str
    blind_id: str  # Random ID for blind evaluation


def load_humaneval_problems(problem_ids: list[str]) -> dict:
    """Load specific HumanEval problems."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = {}
    for item in ds:
        task_id = item["task_id"]
        if task_id in problem_ids:
            problems[task_id] = {
                "prompt": item["prompt"],
                "test": item["test"],
                "entry_point": item["entry_point"],
                "canonical_solution": item["canonical_solution"],
            }
    return problems


def generate_completion(client: anthropic.Anthropic, 
                       prompt: str, 
                       condition: Condition,
                       model: str = "claude-sonnet-4-20250514") -> str:
    """Generate a code completion with the given condition."""
    full_prompt = f"{condition.prefix}Complete the following Python function:\n\n{prompt}{condition.suffix}"
    
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.7,  # Some variation to see effect across samples
    )
    
    # Extract the completion from the response
    content = response.content[0].text
    
    # Try to extract just the function body if wrapped in markdown
    if "```python" in content:
        content = content.split("```python")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    return content.strip()


def check_correctness(prompt: str, completion: str, test: str, entry_point: str, 
                      timeout: float = 5.0) -> tuple[bool, Optional[str]]:
    """Execute the completion against unit tests."""
    # Combine prompt + completion + tests
    full_code = f"{prompt}{completion}\n\n{test}\n\ncheck({entry_point})"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        f.flush()
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr[:500]  # Truncate long errors
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:500]
    finally:
        os.unlink(temp_path)


def run_experiment(
    n_trials_per_condition: int = 5,
    output_dir: str = "./results",
    model: str = "claude-sonnet-4-20250514"
) -> dict:
    """Run the full experiment."""
    
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading HumanEval problems...")
    problems = load_humaneval_problems(SELECTED_PROBLEMS)
    print(f"Loaded {len(problems)} problems")
    
    results = []
    blind_outputs = []  # For blind evaluation later
    
    # Create all trial combinations and shuffle for balanced ordering
    trials = [
        (problem_id, condition)
        for problem_id in problems.keys()
        for condition in CONDITIONS
        for _ in range(n_trials_per_condition)
    ]
    random.shuffle(trials)
    
    print(f"Running {len(trials)} trials...")
    
    for i, (problem_id, condition) in enumerate(trials):
        problem = problems[problem_id]
        print(f"  [{i+1}/{len(trials)}] {problem_id} / {condition.name}")
        
        try:
            completion = generate_completion(
                client, 
                problem["prompt"], 
                condition,
                model=model
            )
            
            passed, error = check_correctness(
                problem["prompt"],
                completion,
                problem["test"],
                problem["entry_point"]
            )
        except Exception as e:
            completion = ""
            passed = False
            error = f"Generation error: {e}"
        
        # Generate blind ID (hash of timestamp + random)
        blind_id = hashlib.sha256(
            f"{datetime.now().isoformat()}{random.random()}".encode()
        ).hexdigest()[:12]
        
        result = TrialResult(
            problem_id=problem_id,
            condition=condition.name,
            completion=completion,
            passed=passed,
            error=error,
            timestamp=datetime.now().isoformat(),
            blind_id=blind_id
        )
        results.append(asdict(result))
        
        # Store blind version (without condition) for later evaluation
        blind_outputs.append({
            "blind_id": blind_id,
            "problem_id": problem_id,
            "prompt": problem["prompt"],
            "completion": completion,
            "passed": passed,
            # condition intentionally omitted for blind eval
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"{output_dir}/results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(f"{output_dir}/blind_eval_{timestamp}.json", "w") as f:
        # Shuffle again for evaluation
        random.shuffle(blind_outputs)
        json.dump(blind_outputs, f, indent=2)
    
    # Key mapping for after blind evaluation
    key_mapping = {r["blind_id"]: r["condition"] for r in results}
    with open(f"{output_dir}/key_{timestamp}.json", "w") as f:
        json.dump(key_mapping, f, indent=2)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PASS RATE BY CONDITION")
    print("="*60)
    
    for condition in CONDITIONS:
        condition_results = [r for r in results if r["condition"] == condition.name]
        passed = sum(1 for r in condition_results if r["passed"])
        total = len(condition_results)
        rate = passed / total if total > 0 else 0
        print(f"  {condition.name:20} {passed:3}/{total:3} = {rate:.1%}")
    
    print("\n" + "="*60)
    print("PASS RATE BY PROBLEM")
    print("="*60)
    
    for problem_id in problems.keys():
        problem_results = [r for r in results if r["problem_id"] == problem_id]
        passed = sum(1 for r in problem_results if r["passed"])
        total = len(problem_results)
        rate = passed / total if total > 0 else 0
        print(f"  {problem_id:20} {passed:3}/{total:3} = {rate:.1%}")
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - results_{timestamp}.json (full data)")
    print(f"  - blind_eval_{timestamp}.json (for blind quality eval)")
    print(f"  - key_{timestamp}.json (condition key, open after eval)")
    
    return {
        "results": results,
        "summary": {
            "total_trials": len(results),
            "overall_pass_rate": sum(1 for r in results if r["passed"]) / len(results),
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Conditioning Test")
    parser.add_argument("-n", "--trials", type=int, default=5,
                       help="Number of trials per condition per problem")
    parser.add_argument("-o", "--output", type=str, default="./results",
                       help="Output directory")
    parser.add_argument("-m", "--model", type=str, default="claude-sonnet-4-20250514",
                       help="Model to test")
    
    args = parser.parse_args()
    
    run_experiment(
        n_trials_per_condition=args.trials,
        output_dir=args.output,
        model=args.model
    )
