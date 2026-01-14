# Semantic Conditioning Test

**Hypothesis:** Rare tokens with high-quality semantic associations (like ἀρετή "excellence" and 職人気質 "craftsman spirit") may activate different attention patterns in LLMs than worn-out phrases like "high quality", potentially yielding better code.

## Rationale

In embedding space, rare tokens are positioned based on their co-occurrence patterns in training data. Words like ἀρετή primarily appear in philosophical discussions of excellence, virtue, and mastery. 職人気質 appears in contexts discussing Japanese craftsmanship traditions. 

By contrast, "high quality" appears everywhere—marketing copy, spam, product descriptions, genuine engineering discussions. The signal is diluted.

The hypothesis: using rare, semantically-rich tokens might "teleport" the model's attention to a different neighborhood of vector space where high-quality outputs are more likely.

## Design

### Tasks
Uses problems from OpenAI's **HumanEval** benchmark:
- 164 programming problems with unit tests
- Objective correctness measurement (pass/fail)
- Standard benchmark = comparable to published results

### Conditions

| Condition | Prompt prefix |
|-----------|--------------|
| control | (none) |
| common_english | "Write high-quality, production-ready code." |
| greek_arete | "Write code embodying ἀρετή (excellence)." |
| japanese_shokunin | "Write code with 職人気質 (craftsman spirit)." |
| combined | "Write code embodying ἀρετή and 職人気質." |

### Measurements

**Objective:** Pass rate on HumanEval unit tests
**Subjective:** Blind quality evaluation on 5 dimensions:
1. Edge case handling (unprompted)
2. Error handling quality
3. Idiomaticity (Pythonic style)
4. Documentation quality
5. "Would I ship this?" gut check

## Running the Experiment

### Prerequisites
```bash
pip install datasets anthropic
export ANTHROPIC_API_KEY=your-key-here
```

### Step 1: Run trials
```bash
# Quick test (5 trials × 5 conditions × 5 problems = 125 API calls)
python semantic_conditioning_test.py -n 5

# More statistical power (10 trials = 250 calls)
python semantic_conditioning_test.py -n 10 -m claude-sonnet-4-20250514
```

This produces:
- `results/results_TIMESTAMP.json` - Full data with conditions
- `results/blind_eval_TIMESTAMP.json` - Shuffled outputs for blind eval
- `results/key_TIMESTAMP.json` - Condition mapping (don't peek!)

### Step 2: Blind evaluation
```bash
python evaluate_quality.py results/blind_eval_TIMESTAMP.json
```

Score each sample on the 5-dimension rubric. Don't look at the key file!

### Step 3: Reveal results
```bash
python reveal_results.py results/results_TIMESTAMP.json \
                         results/scores_TIMESTAMP.json \
                         results/key_TIMESTAMP.json
```

## Interpreting Results

**If semantic conditioning works:**
- `greek_arete` and `japanese_shokunin` should score higher than `common_english`
- `combined` might show additive effect
- Pass rates should be similar (conditioning affects quality, not correctness)

**Alternative explanations to consider:**
- ANY quality instruction helps equally → not about rare tokens
- Evaluator bias → need multiple blind evaluators
- Problem selection → try different HumanEval subsets
- Model sensitivity → test across model families

## Limitations

- Small sample sizes (consider n≥20 for statistical significance)
- Single evaluator introduces noise
- HumanEval problems may be too simple to show quality differentiation
- Temperature setting (0.7) affects variance

## Extensions

1. **More conditions:** Test other rare quality-associated terms
2. **Different tasks:** Apply to prose, documentation, design
3. **Token analysis:** Actually examine attention patterns
4. **Cross-model:** Test if effect transfers across model families
5. **A/B at scale:** Run in production with user quality ratings

## Files

```
semantic_conditioning_test.py  # Main experiment runner
evaluate_quality.py            # Blind evaluation interface  
reveal_results.py              # Post-evaluation analysis
results/                       # Output directory
```
