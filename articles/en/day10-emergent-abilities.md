# Day 10: Emergent Abilities

> **Core Question**: Do large language models suddenly develop new capabilities at certain scales, or is this just a measurement artifact?

---

## Opening

In 2022, researchers at Google published a fascinating observation: certain tasks that small language models utterly failed at would suddenly become solvable when models reached a critical size. Three-digit addition? Zero percent accuracy at 10 billion parameters, then 85% at 100 billion. Word unscrambling? Same pattern.

They called these **emergent abilities**—capabilities that appear abruptly and unpredictably as models scale up.

This finding sent shockwaves through the AI community. If true, it means we can't predict what a model will be capable of just by looking at smaller versions. It means scaling might unlock fundamentally new behaviors, not just better versions of existing ones. It hints at something almost magical happening inside these networks.

But here's the twist: a year later, another group of researchers challenged this narrative. They argued that emergence might be a **mirage**—an artifact of how we measure performance, not a real phenomenon in the models themselves.

Today, we'll dive deep into this debate. What exactly is emergence? What evidence supports it? And what does the counter-argument say? By the end, you'll understand one of the most fascinating and contentious questions in modern AI research.

---

## 1. What Are Emergent Abilities?

### 1.1 The Basic Pattern

**Emergent abilities** are capabilities that:
1. Are **absent** in smaller models (near-random performance)
2. **Suddenly appear** at a specific scale threshold
3. Show **sharp, discontinuous** improvement rather than gradual progress

Think of it like heating water. As temperature rises from 0°C to 99°C, water stays liquid—gradual, predictable changes. Then at 100°C, boom: it suddenly becomes steam. This is a **phase transition**—a qualitative change that happens at a critical threshold.

![Phase Transition Analogy](../zh/images/day10/phase-transition-analogy.png)
*Figure 1: Phase transitions show discontinuous change at critical thresholds. Does AI exhibit similar behavior?*

The emergence hypothesis suggests LLMs undergo something similar. Below a certain compute/parameter threshold, they can't do a task. Above it, they suddenly can.

### 1.2 The Original Evidence

The term "emergent abilities" was popularized by the 2022 BIG-bench paper and subsequent analysis. Researchers tested language models of various sizes on 200+ diverse tasks and found a striking pattern:

![Emergent Abilities Sharp Jump](../zh/images/day10/emergent-abilities-sharp-jump.png)
*Figure 2: Some tasks show sharp performance jumps at specific model scales, while others improve smoothly.*

Key observations:
- **Smooth scaling**: Some capabilities (like perplexity, factual recall) improve gradually with scale
- **Emergent scaling**: Other tasks (like multi-step arithmetic, certain reasoning tasks) show sudden jumps
- **Different thresholds**: Different tasks emerge at different scales—no single "critical mass"

The paper identified several emergent abilities:
- **Three-digit addition** (emerges ~10²² FLOPs)
- **Word unscrambling** (emerges ~10²³ FLOPs)
- **Chain-of-thought reasoning** (shows significant improvement ~100B parameters in few-shot evaluation; smaller models can benefit from CoT with fine-tuning)
- **Multi-task NLU** (multi-task Natural Language Understanding — using a single model to handle diverse NLU tasks like sentiment analysis, QA, textual entailment, and semantic similarity; emerges at varying scales depending on task difficulty)

---

## 2. The Phase Transition Hypothesis

### 2.1 Why Would Emergence Happen?

If emergence is real, why would it occur? Several hypotheses attempt to explain:

**Hypothesis 1: Critical Representation Capacity**

> Related: *"Emergent Abilities of Large Language Models"* — Wei, Tay et al. (2022, [arXiv:2206.07682](https://arxiv.org/abs/2206.07682))

Some tasks require storing and manipulating multiple concepts simultaneously. A model might need enough parameters to:
1. Encode the rules of arithmetic
2. Store intermediate results
3. Apply operations in sequence
4. Format the final output

Below a threshold, the model literally doesn't have enough "mental workspace." Above it, everything clicks into place.

**Hypothesis 2: Compositional Capabilities**

> Related: *"Are Emergent Abilities of Large Language Models a Mirage?"* — Schaeffer, Miranda, Koyejo (2023, [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)); *"Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"* — Power et al. (2022)

Complex abilities might require combining simpler skills. For example, solving "7 × 8 + 3" requires:
- Understanding multiplication
- Understanding addition  
- Knowing the correct order of operations
- Executing multi-step computation

Each sub-skill might exist partially at smaller scales, but the full capability only "emerges" when all components reach sufficient quality simultaneously.

**Hypothesis 3: In-Context Learning Threshold**

> Related: *"An Explanation of In-context Learning"* — Xie et al. (2022, [arXiv:2212.00759](https://arxiv.org/abs/2212.00759)); *"Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"* — Min et al. (2022)

Many emergent abilities involve **in-context learning**—using examples in the prompt to guide behavior. This meta-ability might have its own threshold:

$$
\text{ICL Capability} = f(\text{model scale}, \text{context length}, \text{task complexity})
$$

When in-context learning ability crosses a threshold, a whole family of tasks suddenly becomes solvable.

### 2.2 Connections to Physics

The phase transition analogy isn't just a metaphor—there are deep connections to statistical physics.

In physics, phase transitions occur in systems with many interacting components. At critical points, small changes in temperature (or other parameters) cause large-scale reorganization.

Neural networks have millions or billions of interacting parameters. Could they exhibit similar critical phenomena? Some theoretical work suggests yes:

- **Loss landscape transitions**: The geometry of the loss function may change qualitatively at certain scales
- **Grokking**: Models sometimes suddenly generalize after extended training—another discontinuous transition
- **Lottery ticket hypothesis**: The "winning subnetwork" might require minimum scale to exist.

> **Note:** The Lottery Ticket Hypothesis (Frankle & Carbin, 2019) has been empirically validated for explaining why overparameterized networks train well, but its application to emergent abilities is an analogy, not a directly verified claim. There is indirect evidence that larger networks contain better subnetworks (Ramanujan et al., 2020), but no paper has proven that a specific emergent capability requires a minimum-scale subnetwork.

---

## 3. The Counter-Argument: Emergence as Mirage

### 3.1 The Measurement Artifact Hypothesis

In 2023, a paper by Schaeffer, Miranda, and Koyejo proposed a provocative alternative: **emergent abilities might not exist at all**.

Their argument centers on how we measure performance:

![Measurement Artifact](../zh/images/day10/measurement-artifact.png)
*Figure 3: Different metrics can make the same underlying improvement look either sudden (left) or gradual (right).*

**The key insight**: Many "emergent" tasks use **discontinuous metrics** like accuracy (correct/incorrect), while the underlying model capability improves continuously.

Here's why this matters:

1. **True capability** (hidden): The model's probability of producing the correct answer improves smoothly with scale
2. **Measured accuracy** (observed): Either the answer is right (100%) or wrong (0%)
3. **Result**: Smooth underlying improvement creates a sharp jump in accuracy when the probability crosses 50%

### 3.2 The Three-Digit Addition Example

Consider three-digit addition like "234 + 567 = ?":

| Model Scale | P(correct) | Accuracy |
|-------------|------------|----------|
| 1B params   | 0.05       | 0%       |
| 10B params  | 0.25       | 0%       |
| 50B params  | 0.45       | 0%       |
| 100B params | 0.65       | 65%      |
| 200B params | 0.85       | 85%      |

The underlying probability improves smoothly from 5% to 85%. But accuracy jumps from 0% to 65% because below ~50% probability, the model gets most questions wrong.

If we instead measured **log-probability of the correct answer**, we'd see smooth improvement all along!

### 3.3 Supporting Evidence

Schaeffer et al. provided compelling evidence:

1. **Metric transformation**: When "emergent" tasks were re-evaluated with smooth metrics (log-probability, Brier score), the sharp jumps disappeared

2. **Resolution effects**: With more evaluation samples, the "jump" becomes less sharp—it's actually a steep sigmoid, not a discontinuity

3. **Multiple-choice artifacts**: Tasks with few answer choices (yes/no, A/B/C) are more likely to show apparent emergence due to threshold effects

![BIG-bench Emergence](../zh/images/day10/big-bench-emergence.png)
*Figure 4: BIG-bench results show apparent emergence, but the pattern may depend on metric choice.*

---

## 4. The Evidence on Both Sides

So which view is correct? The debate continues, and the evidence is genuinely mixed.

### 4.1 Evidence FOR Genuine Emergence

**Chain-of-Thought Reasoning**

Some researchers argue that chain-of-thought (CoT) prompting genuinely enables new capabilities, not just better expression of existing ones. In the original few-shot evaluation framework, models below ~100B parameters couldn't produce coherent multi-step reasoning even when prompted to do so. However, this doesn't mean smaller models lack CoT ability entirely—when specifically fine-tuned for step-by-step reasoning, smaller models can also perform well. The "emergence" of CoT is thus closely tied to the evaluation methodology, not just model scale.

**Qualitative Capability Differences**

There are tasks where small models don't just perform poorly—they show no understanding at all. For example:
- GPT-2 (1.5B): Produces incoherent output on analogy tasks
- GPT-3 (175B): Produces sensible, often correct analogies

This seems like more than measurement artifact—it's a qualitative difference in behavior.

**Cross-Metric Consistency**

For some tasks, the jump appears regardless of metric choice. This is harder to explain away as measurement artifact.

![Emergence Debate Evidence](../zh/images/day10/emergence-debate-evidence.png)
*Figure 5: The evidence varies by task—some show jumps even with smooth metrics.*

### 4.2 Evidence AGAINST Genuine Emergence

**Metric Dependence**

The strongest counter-evidence: many "emergent" abilities become smooth when measured differently. This suggests the underlying capability was always improving gradually.

**Sample Size Effects**

With more evaluation examples, sharp transitions become smoother. This suggests we're observing statistical artifacts, not fundamental transitions.

**No Physical Necessity**

Unlike water's phase transition (which arises from fundamental physics), there's no known mechanism that would cause neural networks to exhibit true discontinuities. The architecture and training are continuous.

### 4.3 Current Consensus

The field hasn't reached consensus, but a nuanced view is emerging:

1. **Some apparent emergence is definitely measurement artifact** (explained by metric choice)
2. **Some tasks may show genuine qualitative transitions** (but this is harder to prove)
3. **The threshold for capabilities is predictable in hindsight** (suggests underlying continuity)
4. **Practical importance varies by use case**

---

## 5. Timeline of the Debate

![Emergence Timeline](../zh/images/day10/emergence-timeline.png)
*Figure 6: The emergence debate has evolved rapidly over just a few years.*

**2020**: GPT-3 demonstrates surprising few-shot learning capabilities. The term "emergence" begins circulating informally.

**2022 (January)**: BIG-bench collaboration publishes results showing "emergent" performance patterns across many tasks.

**2022 (March)**: Chain-of-thought prompting paper shows that step-by-step reasoning dramatically improves performance—but only at scale.

**2022 (April)**: PaLM paper explicitly discusses "emergent abilities" and identifies 8 tasks showing this pattern.

**2022 (November)**: ChatGPT launches, sparking massive public interest in what LLMs can (and can't) do.

**2023 (April)**: Schaeffer et al. publish "Are Emergent Abilities of Large Language Models a Mirage?" challenging the emergence narrative.

**2023-2024**: Ongoing research attempts to distinguish true emergence from measurement artifacts. No definitive resolution yet.

---

## 6. Why Does This Matter?

The emergence debate isn't just academic philosophy—it has real practical implications.

![Emergence Implications](../zh/images/day10/emergence-implications.png)
*Figure 7: The answer to "is emergence real?" affects research priorities, safety analysis, and engineering decisions.*

### 6.1 If Emergence is REAL

**Safety implications**: We can't predict what a scaled-up model will do. New, potentially dangerous capabilities might appear suddenly without warning. This argues for extreme caution in scaling.

**Research priority**: Understanding the conditions for emergence becomes crucial. We should invest heavily in mechanistic interpretability to understand what happens at transition points.

**Engineering**: Predicting capability from scale becomes harder. We might need to train and evaluate many model sizes to find the "sweet spot" for a task.

### 6.2 If Emergence is a MIRAGE

**Safety implications**: While scaling still brings risks, capabilities are more predictable. We can extrapolate from smaller models with more confidence.

**Research priority**: Focus on improving underlying capabilities smoothly rather than searching for magic thresholds. Better evaluation metrics become the priority.

**Engineering**: Scaling laws become more reliable for planning. We can predict capabilities before training expensive large models.

### 6.3 The Pragmatic View

Regardless of which view is "true," there are practical lessons:

1. **Use multiple metrics**: Don't rely solely on accuracy. Track log-probabilities, token-level metrics, and calibration.

2. **Test at multiple scales**: Don't assume small model failure predicts large model failure (or success).

3. **Understand your evaluation**: Know what your metrics actually measure and their resolution limits.

4. **Plan for surprises**: Whether true emergence or not, scaled models will have capabilities you didn't explicitly train for.

---

## 7. Mathematical Framework

> *This section provides a more formal treatment for readers interested in the mathematical details.*

### 7.1 Formalizing Emergence

Let $C(N)$ represent the true capability level of a model with $N$ parameters on a specific task. The "emergence as artifact" hypothesis proposes:

$$
\begin{aligned}
C(N) &= f(N) \quad &\text{(smooth function)} \\
\text{Accuracy}(N) &= \mathbb{1}[C(N) > \theta] \quad &\text{(threshold function)}
\end{aligned}
$$

The accuracy metric applies a threshold $\theta$, creating apparent discontinuity even when $f(N)$ is smooth.

### 7.2 The Log-Linear Alternative

If we instead measure log-probability of the correct answer:

$$
\log P(\text{correct} \mid N) = \alpha \log N + \beta
$$

This predicts smooth, predictable improvement—consistent with the "mirage" hypothesis.

### 7.3 Testing for True Emergence

To distinguish true emergence from artifacts, we need:

1. **Resolution analysis**: Does the "jump" survive with more evaluation samples?
2. **Metric invariance**: Does the pattern hold across different metrics?
3. **Mechanistic evidence**: Can we identify what changes in the model at the transition?

Currently, no task has passed all three tests convincingly.

---

## 8. Common Misconceptions

### ❌ "Emergence means we don't understand the model"

**Reality**: Emergence refers to a specific scaling pattern, not a lack of understanding. Emergent capabilities can still be analyzed and understood mechanistically.

### ❌ "If emergence is a mirage, scaling doesn't improve capabilities"

**Reality**: The mirage hypothesis doesn't deny that scaling improves capabilities—it argues the improvement is smooth, not sudden. Capabilities still get dramatically better with scale.

### ❌ "Either emergence is completely real or completely fake"

**Reality**: The truth is likely nuanced. Some observed emergences are measurement artifacts; others might reflect genuine qualitative transitions. The debate is about the proportion and mechanism, not a binary truth.

---

## 9. Further Reading

### Foundational Papers

1. [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) - Wei et al. (2022)
   The paper that coined "emergent abilities" and documented patterns across BIG-bench tasks.

2. [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) - Schaeffer et al. (2023)
   The counter-argument claiming emergence is a measurement artifact.

3. [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615) - BIG-bench Collaboration (2022)
   The massive benchmark that sparked the emergence debate.

### Theoretical Perspectives: Why Does Emergence Happen (or Not)?

The debate over emergent abilities connects to deeper theoretical questions across multiple disciplines. Researchers have approached this from different angles, but no unified theory exists yet. Here are the main threads:

**Statistical Physics**

Statistical mechanics studies how macroscopic behavior emerges from microscopic interactions — a natural analogy for neural networks.

4. [Statistical Mechanics of Deep Learning](https://arxiv.org/abs/2006.06039) — Bahri et al. (2020)
   Maps neural network training to statistical mechanics, including phase transitions.

5. [Phase Transitions in Deep Learning](https://arxiv.org/abs/2202.05763) — Roberts et al. (2022)
   Directly analyzes phase transition behavior in neural networks using statistical physics frameworks.

6. Work by Zdeborová & Krzakala — Systematic application of spin glass theory and statistical physics to overparameterized networks, currently the most rigorous theoretical route.

**Information Theory**

7. [Information Bottleneck method] — Tishby & Zaslavsky (2015)
   Proposes that networks undergo two training phases (fitting then compression), with the compression phase potentially corresponding to a phase transition.

8. Random matrix theory (Pennington & Bahri, 2017) — Shows that eigenvalue distributions of large parameter matrices transition from ordered to chaotic regimes as scale increases.

**Complex Systems**

9. Research at Santa Fe Institute (Melanie Mitchell et al.) — Studies emergence as a general phenomenon in complex systems. Their position: LLM emergence may be fundamentally different from classical emergence in complex systems (e.g., bird flocking, economic systems), but the theoretical tools may still apply.

10. [Renormalization group analysis of deep networks] — Bhattacharya et al. (2024)
    Finds scaling behavior across network layers analogous to physical systems.

**Related Empirical Phenomena**

11. [Grokking: Generalization Beyond Overfitting](https://arxiv.org/abs/2201.06091) — Power et al. (2022)
    Models suddenly generalize after prolonged overfitting — an experimentally observed phase transition.

12. [Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://www.pnas.org/doi/10.1073/pnas.1903070116) — Belkin et al. (2019)
    Documents double descent: test error decreases → increases → decreases again as model scale grows, challenging classical theory.

**The bottom line**: Emergence is an active research frontier. Information theory, statistical physics, and complex systems theory all offer partial explanations, but none has produced a definitive theorem like "when network scale reaches X, capability Y must appear." Whether emergence is real or a mirage, understanding *why* it appears (or appears to appear) is one of the most important open questions in AI theory.

**Latest Breakthrough (2025)**

One of the most compelling pieces of evidence for genuine emergence comes from Zdeborová's group at EPFL:

15. [A phase transition between positional and semantic learning in a solvable model of dot-product attention](https://iopscience.iop.org/article/10.1088/1742-5468/ade137) — Cui et al. (2025, JSTAT)

They proved, using statistical physics methods, that Transformers undergo a genuine phase transition during training: with limited data, the model relies solely on word *positions* to understand sentences; but once training data crosses a critical threshold, the strategy **abruptly shifts** to relying on word *semantics*. This transition is not gradual — it's discontinuous, like a phase transition in physics.

This is perhaps the strongest theoretical evidence to date that genuine phase transitions exist in neural network learning — not just measurement artifacts, but real qualitative shifts in how the model processes information.

> Also see: [Single-Head Attention in High Dimensions](https://arxiv.org/abs/2509.24914) — Boncoraglio, Troiani, Zdeborová, Krzakala (2025) — a theoretical framework for generalization, weight spectra, and scaling laws in attention mechanisms.

**Notable recent development (2025):** A team led by Zdeborová published a rigorous result in JSTAT — *"A phase transition between positional and semantic learning in a solvable model of dot-product attention"* (Cui et al., 2025). Using statistical physics methods, they proved that during Transformer self-attention training, models initially rely on word positions, but once training data crosses a critical threshold, they **abruptly** transition to using word semantics. This is not gradual — it's a genuine phase transition, mathematically proven in a simplified setting. This may be the strongest theoretical support yet for the existence of real emergence in neural networks.

15. [A phase transition between positional and semantic learning in a solvable model of dot-product attention](https://doi.org/10.1088/1742-5468/ade137) — Cui et al. (2025)

### Other Key Resources

13. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - Wei et al. (2022)
   Demonstrates emergent reasoning capabilities through prompting.

14. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al. (2020)
   Foundation for understanding how capabilities scale (see Day 9).

---

## 10. Reflection Questions

1. **If you were designing a safety evaluation for a new scaled-up model, how would the emergence debate affect your approach?**

2. **What would convince you that emergence is real (or a mirage)? What evidence would you need to see?**

3. **How does the emergence debate connect to broader questions about whether AI systems can have qualitatively new capabilities, or only quantitative improvements on existing ones?**

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Emergent Abilities | Capabilities that appear suddenly at specific model scales |
| Phase Transition | Qualitative change at a critical threshold (analogy from physics) |
| Measurement Artifact | Apparent emergence caused by discontinuous metrics |
| Smooth Metrics | Log-probability, Brier score—reveal gradual improvement |
| BIG-bench | 200+ task benchmark that documented emergence patterns |
| The Debate | Ongoing: real phenomenon vs. mirage from metric choice |

**Key Takeaway**: The question "do emergent abilities exist?" doesn't have a simple answer. The evidence suggests some apparent emergence is definitely a measurement artifact (use smooth metrics and the jump disappears), but some tasks may show genuine qualitative transitions. What matters practically is recognizing that scaled-up models will have capabilities you didn't predict—whether that's "emergence" or just the result of smooth improvement we failed to anticipate. Use multiple metrics, test at multiple scales, and stay humble about prediction.

---

*Day 10 of 60 | LLM Fundamentals*  
*Word count: ~3100 | Reading time: ~15 minutes*
