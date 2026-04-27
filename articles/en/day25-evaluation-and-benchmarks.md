# Day 25: Evaluation & Benchmarks

> **Core Question**: How do we know if an LLM is actually good — and why are the most popular benchmarks becoming unreliable?

---

## Opening

Imagine you're shopping for a phone. You see two models: one scored 98 on "PhoneBench" and another scored 92. Easy choice, right? But what if "PhoneBench" only tests how fast the phone can open its own manufacturer's apps? The 98-score phone might be gaming the test.

That's essentially where LLM evaluation finds itself in 2026. The most famous benchmark, MMLU (Massive Multitask Language Understanding), is saturated — top models all score 90%+. HumanEval, the go-to coding benchmark, sees models hitting 95%+. When everyone aces the test, the test stops being useful.

This article is about how we measure LLMs, why measurement is harder than it looks, and what the evaluation landscape looks like when old benchmarks crumble and new ones rise.

![Figure 1: The landscape of LLM evaluation benchmarks organized by category](../zh/images/day25/benchmark-landscape.png)
*Figure 1: The major categories of LLM benchmarks — knowledge & reasoning, coding, and human preference — each with distinct evaluation goals.*

---

## 1. Why Evaluation Matters (and Why It's Hard)

#### Intuition: The Report Card Problem

Think of LLM evaluation like grading students. A multiple-choice test tells you if a student memorized facts, but not if they can think. An essay test reveals thinking, but grading is subjective. A group project shows collaboration, but one person might carry the team.

LLMs face the same problem: no single test captures everything that matters.

### 1.1 What We Want to Measure

At a high level, we evaluate LLMs along several axes:

| Axis | What It Measures | Example Benchmark |
|------|-----------------|-------------------|
| Knowledge | Facts and world knowledge | MMLU, TriviaQA |
| Reasoning | Logical and mathematical thinking | GSM8K, AIME, ARC-AGI |
| Coding | Writing and understanding code | HumanEval, SWE-bench |
| Instruction Following | Doing what the user asked | IFEval |
| Safety | Avoiding harmful outputs | TruthfulQA, ToxiGen |
| Human Preference | What people actually prefer | Chatbot Arena |

The problem? These axes aren't independent. A model that's great at coding might be mediocre at following instructions. A model that's safe might refuse too many reasonable requests. There's no single number that tells you "this model is better."

### 1.2 The Evaluation Pipeline

How does evaluation actually work in practice?

![Figure 2: The evaluation pipeline from benchmark selection to scoring](../zh/images/day25/evaluation-pipeline.png)
*Figure 2: The standard evaluation pipeline — from selecting a benchmark through running inference to scoring results.*

The basic process is:

1. **Select a benchmark** — choose tasks that reflect what you care about
2. **Prepare prompts** — format questions into the model's expected input
3. **Run inference** — generate model outputs (often at temperature 0 for determinism)
4. **Collect outputs** — gather all responses
5. **Score and compare** — compute metrics and rank models

Sounds simple. But each step hides complexity — especially the scoring.

---

## 2. The Major Benchmarks

### 2.1 MMLU — The Knowledge Standard (Now Saturated)

MMLU (Massive Multitask Language Understanding), introduced in 2021 by Hendrycks et al., tests knowledge across 57 subjects — from abstract algebra to veterinary medicine. It's multiple-choice with four options.

**Why it matters**: MMLU was the first comprehensive, multi-domain knowledge test. For years, it was *the* number everyone quoted.

**Why it's fading**: By 2025-2026, frontier models all score 88-94%. When the gap between the #1 and #5 model is 3 percentage points, MMLU stops being useful for comparing them.

The response has been **MMLU-Pro** (harder questions, 10 options instead of 4) and **MMLU-CF** (contamination-free version with rephrased questions). MMLU-Pro causes a 16-33% accuracy drop compared to original MMLU, which tells you how much of the original scores were inflated.

$$
\begin{aligned}
\text{MMLU Accuracy} &= \frac{\text{Correct answers}}{\text{Total questions}} \\
\text{MMLU-Pro Accuracy} &\approx \text{MMLU Accuracy} - 0.20 \text{ to } 0.30
\end{aligned}
$$

### 2.2 GPQA — Graduate-Level Science

GPQA (Google-Proof Question Answering), introduced by Rein et al. in 2023, contains questions so hard that even PhD-level humans with Google access struggle. The "Diamond" subset is the hardest.

**Why it matters**: This is one of the few benchmarks that still clearly differentiates models. As of April 2026, top scores are around 65-75% — far from saturated. If you want to know which model reasons deepest about science, use GPQA Diamond.

### 2.3 HumanEval — Coding Benchmark (Approaching Saturation)

HumanEval, from Chen et al. (2021), gives models function signatures and docstrings, then checks if the generated code passes unit tests. The metric is **pass@k** — the probability that at least one of k generated solutions is correct.

$$
\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

Where n is the total number of samples and c is the number of correct ones.

**Why it matters**: It's simple, objective, and automatable. Code either runs or it doesn't.

**Why it's fading**: The tasks are small standalone functions. Real-world coding involves understanding large codebases, debugging, and multi-file changes. That's where **SWE-bench** comes in — it asks models to resolve real GitHub issues from popular repositories. As of 2026, SWE-bench Verified is the preferred coding benchmark for frontier models.

### 2.4 ARC-AGI — Abstract Reasoning

ARC-AGI (Abstraction and Reasoning Corpus), created by François Chollet, tests fluid intelligence through visual pattern puzzles. Models must infer transformation rules from a few examples and apply them to new inputs.

ARC-AGI 2, released in 2025, made the benchmark harder after the original was partially solved. It's specifically designed to test *generalization*, not memorization — the puzzles are procedurally generated and can't be memorized.

### 2.5 Humanity's Last Exam (HLE)

Released in early 2025, HLE is a collaboration between the Center for AI Safety and Scale AI. It contains over 12,000 expert-crafted questions across 14 domains. The name is provocative but the benchmark is real — it was designed to be the "final exam" that even frontier models struggle with. As of 2026, top models score well below 50%, making it one of the best differentiating benchmarks.

### 2.6 Chatbot Arena — Human Preference at Scale

Chatbot Arena, run by LMSYS, takes a completely different approach: real humans have conversations with two anonymous models side-by-side and vote for which response they prefer. The results are aggregated into **Elo ratings**, the same system used in chess.

**Why it matters**: It's the closest thing to "which model do people actually prefer?" — not "which model scores higher on a specific test?" As of 2026, the Arena has collected over 6 million votes and is widely considered the most reliable overall quality signal.

**How Elo works**: Each "battle" between two models is a user vote. If a lower-rated model beats a higher-rated one, it gains more points than if it beats an equal-rated model — exactly like chess. The update rule is:

$$
R_{\text{new}} = R_{\text{old}} + K \cdot (S - E)
$$

Where R is the rating, K is a sensitivity constant (typically 32), S is the actual outcome (1 for win, 0 for loss, 0.5 for tie), and E is the expected score based on current ratings:

$$
E = \frac{1}{1 + 10^{(R_{\text{opponent}} - R_{\text{model}}) / 400}}
$$

As of April 2026, top Arena Elo scores exceed 1400. Claude Opus 4.6 broke 1560 on the coding-specific leaderboard, the first model to surpass 1500.

**Limitations**: Human preferences can be noisy. Users may prefer longer, more confident responses even when they're wrong. The user base may not represent your specific use case.

---

## 3. The Benchmark Saturation Problem

![Figure 3: How benchmarks saturate over time as models improve](../zh/images/day25/benchmark-saturation-timeline.png)
*Figure 3: Benchmark saturation timeline — MMLU and HumanEval are effectively saturated, while GPQA Diamond still differentiates models.*

This is one of the most important dynamics in LLM evaluation. Here's the pattern:

1. A new benchmark is introduced → models perform poorly
2. Model builders optimize for it → scores rise rapidly
3. Scores plateau near ceiling → benchmark no longer differentiates
4. A harder benchmark is created → the cycle repeats

**MMLU** went from ~25% (random chance on 4 options is 25%) for GPT-2 to 90%+ for frontier models. **HumanEval** went from ~10% for early Codex to 95%+. Each time, the benchmark became less useful for ranking models.

#### What This Means in Practice

When you see a model claiming "state-of-the-art on MMLU," ask:
- *What about MMLU-Pro?* (probably much lower)
- *What about contaminated vs. clean splits?* (some questions may be in training data)
- *What does Chatbot Arena say?* (human preference is harder to game)

### 2.7 Why No Single Benchmark Suffices

Different benchmarks test different things. A model can dominate MMLU while struggling with ARC-AGI. This is why practitioners increasingly use **benchmark profiles** — radar charts showing performance across multiple benchmarks simultaneously.

![Figure 3b: Radar chart comparing two models across seven benchmarks](../zh/images/day25/benchmark-radar-comparison.png)
*Figure 3b: Two hypothetical models compared across seven benchmarks. Model A dominates MMLU and HumanEval (saturated), but the gap on SWE-bench and HLE reveals real differences.*

The key insight: **look at the shape, not any single number**. A model with high MMLU but low SWE-bench is a knowledge engine that can't code. A model with high Arena Elo but low GPQA is probably likable but shallow on reasoning.

---

## 4. Data Contamination — The Elephant in the Room

![Figure 4: How benchmark data leaks into training sets](../zh/images/day25/contamination-problem.png)
*Figure 4: Data contamination occurs when benchmark questions appear in the training corpus, inflating scores through memorization rather than genuine capability.*

This is the most serious threat to benchmark validity. Here's how it happens:

### 4.1 How Contamination Occurs

LLMs are trained on massive web scrapes. Benchmarks like MMLU and HumanEval are publicly available on the internet. If benchmark questions (or very similar content) appears in the training data, the model can memorize answers rather than reasoning about them.

A 2025 survey by Gema et al. found that contamination detection methods consistently find overlap between popular benchmarks and common pre-training datasets. The paper "Are We Done with MMLU?" documented significant test-set leakage across multiple commercial models.

### 4.2 Why It's Hard to Detect

Contamination isn't always exact matching. The training data might contain:
- **Paraphrased versions** of benchmark questions
- **Discussion of benchmark answers** in forums and tutorials
- **Similar but not identical** questions from the same source material

Modern contamination detection uses LLMs themselves to check if a model "recognizes" test questions it shouldn't have seen, but this is an arms race.

### 4.3 Mitigation Strategies

| Strategy | How It Works | Trade-off |
|----------|-------------|-----------|
| Hold-out benchmarks | Keep some benchmarks private | Limits community access |
| Dynamic benchmarks | Generate new questions regularly | Expensive to maintain |
| Rephrased versions | Rewrite existing questions | May change difficulty |
| Contamination detection | Check training data for overlap | Can't catch paraphrases |
| Live benchmarks | Use real-time tasks (SWE-bench) | Harder to standardize |

### 4.4 Key Papers

- ["Are We Done with MMLU?"](https://arxiv.org/abs/2406.04127) — documents MMLU contamination issues
- ["A Survey on Data Contamination for Large Language Models"](https://arxiv.org/abs/2502.14425) — comprehensive 2025 survey
- ["When Benchmarks Leak: Inference-Time Decontamination for LLMs"](https://arxiv.org/abs/2601.19334) — January 2026, proposes real-time decontamination

---

## 5. Modern Evaluation: What to Use in 2026

Given saturation and contamination, what should you actually use? Here's a practical guide:

### 5.1 Benchmark Selection Guide

| Your Question | Best Benchmark | Why |
|---------------|---------------|-----|
| Which model is best overall? | Chatbot Arena Elo | Human preference, hard to game |
| Which reasons best about science? | GPQA Diamond | Still unsaturated, expert-level |
| Which is best at real coding? | SWE-bench Verified | Real GitHub issues, not toy functions |
| Which handles hard math? | AIME 2025 | Olympiad-level, no memorization |
| Which is smartest generally? | ARC-AGI 2 | Tests fluid intelligence |
| Which follows instructions? | IFEval | Direct instruction-following test |
| What's the hardest frontier? | HLE | Expert-level, very low scores |

### 5.2 The AAII Index

The Artificial Analysis Intelligence Index (AAII) v3 aggregates 10 challenging evaluations: MMLU-Pro, HLE, GPQA Diamond, AIME, and others into a single composite score. While no single number captures everything, AAII provides a reasonable summary of frontier model capability.

### 5.3 Domain-Specific Evaluation

General benchmarks don't tell you how a model performs on *your* task. For production use:

1. **Build your own eval set** — collect real user queries with gold answers
2. **Use LLM-as-judge** — have a strong model rate outputs on your criteria
3. **A/B test with real users** — the ultimate evaluation
4. **Track drift over time** — models can get worse on your specific use case even as benchmarks improve

---

## 6. Common Misconceptions

### ❌ "A model that scores higher on MMLU is smarter"

Not necessarily. MMLU is saturated and potentially contaminated. A 2% difference on MMLU is noise, not signal. Look at GPQA, ARC-AGI, or Arena Elo instead.

### ❌ "Benchmark scores predict real-world performance"

The gap between benchmark performance and production performance is well-documented. Models that dominate leaderboards often underperform in real deployments because benchmarks test narrow, well-defined tasks while real usage is messy and open-ended.

### ❌ "We just need harder benchmarks"

Harder benchmarks help temporarily, but the saturation cycle repeats. The real solution is **evaluating on your actual use case** rather than relying on any universal benchmark.

### ❌ "Chatbot Arena is perfect because it uses real humans"

Human preference is noisy, biased toward verbosity and confidence, and may not reflect your specific needs. Arena Elo is the best general signal we have, but it's still imperfect.

---

## 7. Code Example: Running a Simple Evaluation

Here's how to run a basic MMLU-style evaluation using Hugging Face datasets:

```python
"""
Simple MMLU-style evaluation using Hugging Face datasets.
Runs multiple-choice questions through a model and computes accuracy.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a small subset of MMLU (STEM subjects)
dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
dataset = dataset.filter(lambda x: x["subject"] in ["abstract_algebra", "astronomy"])

# Load model and tokenizer
model_name = "gpt2"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

def format_mmlu_prompt(question, choices):
    """Format a multiple-choice question as a prompt."""
    labels = ["A", "B", "C", "D"]
    options = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return f"{question}\n{options}\nAnswer:"

def evaluate_one(question, choices, answer_idx):
    """
    Evaluate a single question by comparing log-probabilities
    of each answer choice.
    """
    prompt = format_mmlu_prompt(question, choices)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token's logits

    # Compare log-probabilities of A, B, C, D tokens
    label_tokens = [tokenizer.encode(l)[0] for l in ["A", "B", "C", "D"]]
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = [log_probs[t].item() for t in label_tokens]

    predicted = scores.index(max(scores))
    return predicted == answer_idx

# Run evaluation
correct = 0
total = 0
for example in dataset.select(range(min(50, len(dataset)))):
    if evaluate_one(example["question"], example["choices"], example["answer"]):
        correct += 1
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
```

This shows the basic pattern: format questions, get model scores, compare predictions to answers. Production evaluation systems add prompt engineering, few-shot examples, chain-of-thought, and more sophisticated scoring — but the core loop is the same.

---

## 8. Further Reading

### Beginner
1. [LMSYS Chatbot Arena](https://chat.lmsys.org) — Try it yourself, vote on model comparisons
2. [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) — Community benchmark tracker
3. [LLM Benchmarks Compared (LXT, 2026)](https://www.lxt.ai/blog/llm-benchmarks/) — Good overview of current landscape

### Advanced
1. ["A Survey on Data Contamination for Large Language Models"](https://arxiv.org/abs/2502.14425) — Comprehensive survey on contamination
2. ["Are We Done with MMLU?"](https://arxiv.org/abs/2406.04127) — Analysis of MMLU's limitations
3. ["When Benchmarks Leak: Inference-Time Decontamination for LLMs"](https://arxiv.org/abs/2601.19334) — January 2026 approach to contamination

### Key Papers
1. ["Measuring Massive Multitask Language Understanding" (MMLU)](https://arxiv.org/abs/2009.03300) — Hendrycks et al., 2021
2. ["Evaluating Large Language Models Trained on Code" (HumanEval)](https://arxiv.org/abs/2107.03374) — Chen et al., 2021
3. ["Google-Proof Question Answering" (GPQA)](https://arxiv.org/abs/2311.12022) — Rein et al., 2023
4. ["Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference"](https://arxiv.org/abs/2403.04132) — Zheng et al., 2024
5. ["SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"](https://arxiv.org/abs/2310.06770) — Jimenez et al., 2023

---

## Reflection Questions

1. If you were building a customer service chatbot, which benchmarks would you trust to select the right model — and why wouldn't MMLU be enough?
2. Why do you think benchmark saturation happens so quickly? Is it because benchmarks are poorly designed, or because the field moves fast?
3. How would you design an evaluation program for an LLM product that avoids the pitfalls discussed in this article?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| MMLU | Multi-subject knowledge test, now saturated at 90%+ |
| GPQA Diamond | Hard graduate-level science questions, still differentiates models |
| HumanEval | Function-level coding test, approaching saturation |
| SWE-bench | Real GitHub issue resolution, better for modern coding eval |
| ARC-AGI 2 | Visual pattern reasoning that tests generalization, not memorization |
| HLE (Humanity's Last Exam) | Expert-level questions across 14 domains, very low scores |
| Chatbot Arena | Human preference voting with Elo ratings, 6M+ votes |
| Data Contamination | Benchmark questions leaking into training data, inflating scores |
| Benchmark Saturation | When all top models score similarly, the benchmark stops being useful |

**Key Takeaway**: LLM evaluation is an arms race between benchmark creators and model builders. Old benchmarks saturate, data contamination undermines validity, and no single number captures real-world performance. The best approach in 2026 is to use multiple complementary benchmarks (GPQA, SWE-bench, Arena Elo) and always evaluate on your own specific use case.

---

*Day 25 of 60 | LLM Fundamentals*
*Word count: ~2400 | Reading time: ~12 minutes*
