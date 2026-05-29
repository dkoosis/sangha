# Findings — mini probe (2026-05-29)

A quick, underpowered probe of the semantic-conditioning hypothesis, run because the
full HumanEval × 5-condition × 5-trial experiment was never affordable. Scripts:
`mini_run.py` (pass/fail across conditions) and `mini_style.py` (code-style comparison).

**Setup:** model `claude-haiku-4-5`, 8 inline problems, 5 conditions
(control / generic "high-quality" / ἀρετή / 職人気質 / combined), **1 sample per cell**.

## Result — flat null

| Signal | Result |
|--------|--------|
| Pass rate across all 5 conditions | **7/8, identical** — zero spread |
| Code style on a simple problem (`truncate_number`) | **byte-for-byte identical** output across all 5 conditions (131 chars each) |

The lone failure was the same problem in every condition, and it was a *test-spec bug*
(asserted median 15.0 where the correct answer is 8.0) — the model was right. That it
failed identically across conditions reinforces the null: the prompt prefix changed nothing.

## Interpretation

The mini run does **not kill** the hypothesis, but bounds it:

1. **Correctness (pass rate) is prompt-flavor-blind** and saturated on easy problems —
   the model collapses to the canonical solution regardless of incantation.
2. Any real effect, if it exists, is **too small to see without statistical power**
   (need many repeats to average out temperature noise) **and invisible on tasks with a
   single right answer** — it could only surface on open-ended problems where many valid
   implementations compete, leaving room for a register shift (more guard clauses, docs).

Standing bet: ≈zero on correctness, at most a faint register shift on *subjective* quality —
a few percent buried under noise. Exactly why the full test needed volume + harder problems,
and exactly why it was expensive.

## Mechanism critique

The "rare token teleports attention to a better neighborhood of embedding space" story is
the weak link: ἀρετή is tokenized into subword pieces, and a decoder transformer doesn't
generate "from the prompt's location in embedding space." The poetic intuition isn't the
mechanism. The *outcome* it chases has thin published support (EmotionPrompt, OPRO's
"take a deep breath"), but those effects are small, noisy, and task-dependent.
