#!/usr/bin/env python3
"""
Semantic Conditioning Test: Can rare vocabulary anchors shift code quality/style?

Hypothesis: Rare tokens like "ἀρετή" (Greek: excellence/virtue) may activate
different attention patterns than overused phrases like "high quality",
selecting different evaluative regimes and yielding measurably different code.

Key insight: Anchors don't optimize for "goodness" - they bias tradeoffs.
Different anchors select different value systems the model has learned.

Design:
- Use HumanEval benchmark problems (objective correctness via unit tests)
- Compare conditions: control, generic, strong conventional, rare tokens
- Measure: pass rate (objective) + blind quality evaluation (subjective)

Anchor taxonomy (for future experiments):
- Class A: Norm-dense single tokens (highest efficiency)
- Class B: Compact aphorisms (higher precision)
- Class C: Negative anchors (suppress bad modes)
"""

import json
import os
import random
import hashlib
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

# You'll need: pip install datasets anthropic
from datasets import load_dataset
import anthropic


@dataclass
class Condition:
    name: str
    prefix: str  # Added before the prompt
    suffix: str  # Added after the prompt
    regime: str = "craft"  # Which operating mode this targets
    notes: str = ""  # Rationale for this condition


# =============================================================================
# ANCHOR BANK: Candidate anchors for current and future experiments
# Organized by regime (operating mode) they're hypothesized to select
# =============================================================================

ANCHOR_BANK = {
    # CRAFT / EXCELLENCE REGIME
    # Biases: correctness > novelty, restraint > expressiveness, clarity > surprise
    "craft": {
        "single_tokens": {
            "arete": ("ἀρετή", "excellence/virtue - Greek virtue ethics"),
            "shokunin": ("職人気質", "craftsman spirit - Japanese craft tradition"),
            "akribeia": ("ἀκρίβεια", "precision, exactness - biases toward careful edge handling"),
            "spoude": ("σπουδή", "earnest effort - suppresses glib responses"),
            "gravitas": ("gravitas", "seriousness of purpose - good for architecture"),
            "maestria": ("maestria", "mastery through practice"),
            "finesse": ("finesse", "economical correctness"),
            "parsimony": ("parsimony", "restraint, no excess - good for refactors"),
            "orthogonality": ("orthogonality", "independence of concerns"),
            "coherence": ("coherence", "internal consistency"),
            "integrity": ("integrity", "parts fit the whole"),
        },
        "aphorisms": [
            "Make it correct, then make it clear.",
            "Simple is not easy.",
            "Elegance is achieved by subtraction.",
            "Do not generalize prematurely.",
            "Sharp tools, careful hands.",
            "Every abstraction has a cost.",
            "Leave the codebase calmer than you found it.",
            "Act as if another master will read this.",
            "Write it as if you will maintain it for years.",
        ],
        "negative": [
            "No cleverness.",
            "Avoid ornamental abstraction.",
            "No heroics.",
            "No hand-waving.",
        ],
    },

    # INNOVATION REGIME
    # Biases: novelty + usefulness, risk-tolerant, exploration over optimization
    "innovation": {
        "single_tokens": {
            "shoshin": ("初心", "Beginner's Mind - discard standard patterns"),
            "tabula_rasa": ("tabula rasa", "blank slate - first principles thinking"),
            "promethean": ("promethean", "boldly creative, defying convention"),
            "heuristic": ("heuristic", "signals 'search, don't optimize'"),
            "adjacent_possible": ("adjacent possible", "primes recombination"),
            "speculative": ("speculative", "licenses risk"),
            "generative": ("generative", "favors idea production over pruning"),
        },
        "aphorisms": [
            "Explore before you optimize.",
            "Assume the first solution is wrong.",
            "What would this look like if the constraints were different?",
        ],
        "negative": [],
    },

    # CREATIVITY REGIME
    # Biases: expressiveness + divergence, many options, unusual framing
    "creativity": {
        "single_tokens": {
            "poiesis": ("ποίησις", "bringing-into-being - code as literature"),
            "bricolage": ("bricolage", "tinkering, resourceful hacks"),
            "playful": ("playful", "openness to unusual solutions"),
            "improvisational": ("improvisational", "adaptive, responsive"),
            "expressive": ("expressive", "voice over neutrality"),
            "surreal": ("surreal", "surprisingly effective even in tech"),
        },
        "aphorisms": [
            "Generate, don't judge.",
            "Surprise me.",
            "What's the weird but workable version?",
        ],
        "negative": [],
    },

    # JOY / DELIGHT REGIME
    # Biases: warmth, human resonance, UX focus
    "joy": {
        "single_tokens": {
            "ludic": ("ludic", "playful, game-like"),
            "joie_de_vivre": ("joie de vivre", "joy of living - friendly errors"),
            "eudaemonia": ("eudaemonia", "human flourishing - UX focus"),
            "delight": ("delight", "pleasure in use"),
            "warmth": ("warmth", "human connection"),
            "generosity": ("generosity", "especially good for explanations"),
        },
        "aphorisms": [
            "Make it a pleasure to read.",
            "Assume goodwill.",
            "Write like you want to be understood.",
        ],
        "negative": [],
    },

    # SPEED / MOMENTUM REGIME
    # Biases: throughput > polish, decisiveness, heuristics over proofs
    "speed": {
        "single_tokens": {
            "alacrity": ("alacrity", "eager speed, skip ceremony"),
            "celerity": ("celerity", "swiftness"),
            "expedient": ("expedient", "practical efficiency"),
            "pragmatic": ("pragmatic", "results-focused"),
            "decisive": ("decisive", "action over deliberation"),
            "rough_cut": ("rough-cut", "good enough for now"),
        },
        "aphorisms": [
            "Bias to action.",
            "Good enough for now.",
            "Optimize for iteration speed.",
            "Done is better than perfect.",
        ],
        "negative": [],
    },
}


# =============================================================================
# ACTIVE CONDITIONS: What we're testing in this experiment
# =============================================================================

CONDITIONS = [
    Condition(
        name="control",
        prefix="",
        suffix="",
        regime="baseline",
        notes="No priming - baseline behavior"
    ),
    Condition(
        name="generic_quality",
        prefix="Make this code high quality.\n\n",
        suffix="",
        regime="craft",
        notes="Weak/vague quality instruction - tests if ANY instruction helps"
    ),
    Condition(
        name="common_english",
        prefix="Write high-quality, production-ready code.\n\n",
        suffix="",
        regime="craft",
        notes="Strong conventional instruction - common but specific"
    ),
    Condition(
        name="greek_arete",
        prefix="Write code embodying ἀρετή (excellence).\n\n",
        suffix="",
        regime="craft",
        notes="Rare token from Greek virtue ethics"
    ),
    Condition(
        name="japanese_shokunin",
        prefix="Write code with 職人気質 (craftsman spirit).\n\n",
        suffix="",
        regime="craft",
        notes="Rare token from Japanese craft tradition"
    ),
    Condition(
        name="combined",
        prefix="Write code embodying ἀρετή and 職人気質.\n\n",
        suffix="",
        regime="craft",
        notes="Both rare tokens combined"
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

# Extended set for more statistical power
EXTENDED_PROBLEMS = [
    "HumanEval/2", "HumanEval/4", "HumanEval/11", "HumanEval/22", "HumanEval/29",
    "HumanEval/3",   # below_zero - running sum
    "HumanEval/7",   # filter_by_substring
    "HumanEval/8",   # sum_product
    "HumanEval/10",  # make_palindrome
    "HumanEval/12",  # longest common subsequence
    "HumanEval/18",  # how_many_times substring appears
    "HumanEval/20",  # find_closest_elements
    "HumanEval/23",  # strlen
    "HumanEval/28",  # concatenate
    "HumanEval/31",  # is_prime
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
                       model: str = "claude-sonnet-4-20250514",
                       temperature: float = 0.7) -> str:
    """Generate a code completion with the given condition."""
    full_prompt = f"{condition.prefix}Complete the following Python function:\n\n{prompt}{condition.suffix}"

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
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
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    seed: Optional[int] = None,
    problem_set: str = "selected",
) -> dict:
    """Run the full experiment.

    Args:
        n_trials_per_condition: Number of trials per condition per problem
        output_dir: Where to save results
        model: Model to test
        temperature: Sampling temperature (0.2 for low-variance, 0.7 for diversity)
        seed: Random seed for reproducibility (None = random)
        problem_set: "selected" (5 curated) or "extended" (15 problems)
    """
    # Validate API key early
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
        print(f"Random seed: {seed}")

    client = anthropic.Anthropic()

    Path(output_dir).mkdir(exist_ok=True)

    # Select problem set
    problem_ids = EXTENDED_PROBLEMS if problem_set == "extended" else SELECTED_PROBLEMS
    print(f"Loading HumanEval problems ({problem_set} set, n={len(problem_ids)})...")
    problems = load_humaneval_problems(problem_ids)
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
                model=model,
                temperature=temperature,
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
    
    # Save results with experiment metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "timestamp": timestamp,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "problem_set": problem_set,
        "n_trials_per_condition": n_trials_per_condition,
        "conditions": [c.name for c in CONDITIONS],
    }

    with open(f"{output_dir}/results_{timestamp}.json", "w") as f:
        json.dump({"metadata": metadata, "trials": results}, f, indent=2)
    
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

    parser = argparse.ArgumentParser(
        description="Semantic Conditioning Test: Do rare vocabulary anchors shift code quality?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default settings
  python semantic_conditioning_test.py

  # Low-variance run for cleaner signal
  python semantic_conditioning_test.py --temperature 0.2 --seed 42

  # Full run with extended problem set
  python semantic_conditioning_test.py --problems extended --trials 3

  # Compare temperatures
  python semantic_conditioning_test.py -t 0.2 -o results/low_temp --seed 42
  python semantic_conditioning_test.py -t 0.7 -o results/high_temp --seed 42
"""
    )
    parser.add_argument("-n", "--trials", type=int, default=5,
                        help="Number of trials per condition per problem (default: 5)")
    parser.add_argument("-o", "--output", type=str, default="./results",
                        help="Output directory (default: ./results)")
    parser.add_argument("-m", "--model", type=str, default="claude-sonnet-4-20250514",
                        help="Model to test (default: claude-sonnet-4-20250514)")
    parser.add_argument("-t", "--temperature", type=float, default=0.7,
                        help="Sampling temperature: 0.2 for low-variance, 0.7 for diversity (default: 0.7)")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None = random)")
    parser.add_argument("-p", "--problems", type=str, default="selected",
                        choices=["selected", "extended"],
                        help="Problem set: 'selected' (5 curated) or 'extended' (15 problems)")

    args = parser.parse_args()

    run_experiment(
        n_trials_per_condition=args.trials,
        output_dir=args.output,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        problem_set=args.problems,
    )
