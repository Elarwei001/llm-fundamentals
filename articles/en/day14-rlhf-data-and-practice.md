# Day 14: RLHF Data & Practice

> **Core Question**: How do you actually build the datasets that make RLHF work — and how much does it cost?

---

## Opening

Yesterday we explored the three-stage RLHF pipeline: Supervised Fine-Tuning (SFT), Reward Model training, and PPO optimization. But here's the uncomfortable truth that most papers gloss over: **the algorithm is the easy part. The data is where projects live or die.**

Think of RLHF like cooking a fancy recipe. The algorithm is the oven temperature and cooking time — important, but straightforward. The data is the quality of your ingredients. No amount of precise temperature control will save a dish made with rotten fish.

OpenAI reportedly spent months and millions of dollars on human annotation for InstructGPT. Meta employed a team of annotators for Llama 2's alignment. The companies that won the RLHF game didn't win because they had better algorithms — they won because they had better data pipelines.

Today, we'll dive into the practical side: how to collect SFT demonstrations, how preference annotation actually works, what it costs, and how to measure quality.

---

## 1. The Three Types of RLHF Data

RLHF requires three distinct types of data, each serving a different purpose:

![Figure 1: The Three Types of RLHF Data](../zh/images/day14/rlhf-data-pipeline.png)
*Caption: Each RLHF stage requires different data types with different collection strategies and cost profiles*

### 1.1 SFT Demonstration Data

Supervised Fine-Tuning (SFT) data consists of **prompt-response pairs** where a human demonstrator writes the ideal response. This teaches the model *how* to respond — tone, format, length, and helpfulness.

**Key characteristics:**
- **Volume**: 10K–50K examples (surprisingly small!)
- **Quality bar**: Extremely high — each example is hand-crafted
- **Diversity**: Must cover the full range of intended use cases
- **Cost**: $50K–$500K depending on complexity

The OpenAI InstructGPT paper used roughly 13K SFT examples. Anthropic's Constitutional AI paper used a similar order of magnitude. The insight: **a small amount of high-quality demonstration data goes a long way**. It's better to have 10K excellent examples than 100K mediocre ones.

### 1.2 Preference Comparison Data

This is the core of RLHF — pairs of responses where humans indicate which is better. Think of it as a giant tournament: model generates responses, humans judge them, and we learn a reward signal from these judgments.

**Key characteristics:**
- **Volume**: 100K–1M comparisons (this is the bottleneck)
- **Quality bar**: Moderate per comparison, but consistency matters enormously
- **Annotation type**: Ranking (not scoring — humans are bad at absolute scales)
- **Cost**: $100K–$2M

### 1.3 Prompt Data for RL Training

During the PPO phase, you need a diverse pool of prompts for the model to generate responses and receive rewards. No ground-truth responses are needed — just prompts.

**Key characteristics:**
- **Volume**: 50K–500K prompts
- **Quality bar**: Diversity matters more than quality
- **Cost**: Relatively low — can be synthetically generated or scraped

---

## 2. Preference Annotation: The Bottleneck

Let's zoom into the hardest part: collecting preference comparisons.

![Figure 2: Preference Annotation Workflow](../zh/images/day14/preference-annotation-workflow.png)
*Caption: The end-to-end workflow for collecting preference data, from prompt pool to quality-checked dataset*

### 2.1 Building the Prompt Pool

Before any model generates a single response, you need a **prompt pool** — the set of inputs that will drive your entire annotation pipeline. This isn't just "grab a bunch of questions"; the composition of your prompt pool determines what your reward model learns to optimize.

**Task-specific prompts** are designed to cover your target use cases. If you're building a coding assistant, your pool should include debugging questions, code explanation requests, architecture design problems, and "fix my code" prompts. If you're building a general chatbot, you need a wide spread: creative writing, factual Q&A, math problems, emotional support conversations, and more. The key principle: **your reward model can only learn to judge what it sees during training**. If your prompt pool is all about cooking recipes, don't be surprised when the model is terrible at coding.

**Difficulty levels** matter more than most people realize. You want a deliberate mix of Easy, Medium, and Hard prompts:
- Easy prompts ("What is the capital of France?") test whether the model handles straightforward requests cleanly.
- Medium prompts ("Explain quantum computing to a 10-year-old") test explanatory ability and audience awareness.
- Hard prompts ("Debug this race condition in my distributed system") test deep reasoning and domain expertise.

If all prompts are easy, your reward model never learns to distinguish good from great on complex tasks. If all prompts are hard, it never learns what a solid basic interaction looks like. A balanced mix (e.g., 30% easy, 50% medium, 20% hard) ensures robustness across the full difficulty spectrum.

**Edge cases** are the unsung heroes of a good prompt pool. These include:
- Adversarial prompts designed to trick the model ("Tell me how to hack into...")
- Ambiguous questions where the "right" answer depends on interpretation
- Multi-turn conversations that test context retention
- Prompts in different languages or mixing languages
- Prompts that probe safety boundaries without being explicitly harmful

Why do these matter? Because real users will throw *all kinds of things* at your model. If your reward model has never seen an adversarial prompt during training, it has no basis for judging whether a response handles one well.

> **Analogy**: Think of the prompt pool as a university exam. If you only test with multiple-choice questions, students can pass without understanding anything deeply. A good exam mixes formats — multiple choice, short answer, essay, and trick questions — to truly measure competence. Your prompt pool should do the same.

### 2.2 Response Generation Strategy

Once you have your prompt pool, the next question is: **how do you generate the 2–4 responses that annotators will compare?** This step is more nuanced than it sounds.

**Temperature sampling** is the primary lever for generating diverse responses. Temperature controls the randomness of the model's output:
- Low temperature (T ≈ 0.1–0.3): Nearly deterministic. The model always picks the highest-probability next token. Great for factual tasks, but responses tend to be generic.
- High temperature (T ≈ 1.2–1.5): Very random. The model takes creative chances, sometimes producing surprising and interesting outputs — but sometimes producing nonsense.
- Moderate temperature (T ≈ 0.7–1.0): The sweet spot for RLHF response generation. You get varied but reasonable responses.

Why does this matter? If you generate all 4 responses at T=0.1, they'll be nearly identical — annotators won't have meaningful differences to judge. If all at T=1.5, you'll get gibberish. Moderate temperatures with slight variation (e.g., T = 0.7, 0.8, 0.9, 1.0) give annotators genuinely different but comparable options.

**Different checkpoints** add another dimension of diversity. Your SFT model at epoch 5 and epoch 10 produce responses with noticeably different styles — one might be more concise, the other more verbose. One might follow instructions more faithfully, the other might be more creative. Including responses from different training stages expands the variety in your comparison pool, which leads to a richer reward signal.

**Why 2–4 responses specifically?** This is a practical trade-off:
- Fewer than 2: No comparison is possible — you need at least a pair.
- 2 responses: Minimal viable comparison, but limited signal (only one pairwise comparison per prompt).
- 3–4 responses: Good balance — 4 responses give you 6 pairwise comparisons, which is what the InstructGPT paper used.
- 5+ responses: Diminishing returns and annotator fatigue. Comparing 5 responses meaningfully requires significantly more cognitive effort, and the additional signal per extra response drops sharply.

> **Key insight**: The InstructGPT paper used 4 responses per prompt, which has become something of an industry standard. But even 2 responses per prompt can work well if you have enough prompts. The total number of *comparisons* matters more than the number of responses per prompt.


### 2.3 Why Ranking, Not Scoring?

A natural question: why not just ask annotators to score each response on a scale of 1–5?

It turns out humans are terrible at absolute scoring. If you show the same response to two annotators and ask for a score, you'll get wildly different numbers. But if you show them two responses and ask "which is better?", agreement goes up significantly.

This is a well-known finding from psychophysics (the study of how humans perceive physical stimuli). Humans are much better at **comparative** judgments than **absolute** ones. The Weber-Fechner law tells us that we perceive differences relative to a baseline, not in absolute terms.

> **What is the Weber-Fechner Law?** It's a psychophysical principle stating that perceived difference is proportional to stimulus intensity: ΔI ∝ I. In plain terms: humans perceive changes *relative* to what's already there, not in absolute terms. This is why you can easily tell the difference between a 1kg and 2kg weight, but struggle to distinguish between 50kg and 51kg — the *relative* change matters more than the *absolute* one. Applied to annotation: when two responses are very different in quality, humans easily agree on which is better. But when the difference is subtle (e.g., "not bad" vs. "pretty good"), judgments become highly inconsistent. This is precisely why **ranking (which is better?)** produces more reliable data than **scoring (rate this 1-5)** — comparative judgment sidesteps the absolute-scaling problem that the Weber-Fechner law describes.

In practice, most RLHF datasets use one of these formats:

| Format | Example | Pros | Cons |
|--------|---------|------|------|
| Binary choice | "A or B?" | Simple, fast | Limited signal |
| Ranked list | "Rank A, B, C, D" | Rich comparisons | More cognitive load |
| Likert + pairwise | Score each, then compare | Calibration data | More expensive |

> **What is the "tie" option?** This refers to a strategy used in pairwise comparisons. Typically annotators choose "A better than B" or "B better than A." But when two responses are very similar in quality, annotators hesitate — forcing a choice may be inaccurate. A third "tie" (equal) option allows annotators to honestly say "both are equally good." This avoids noisy data from forced choices and helps the model learn boundary cases where there's no clear preference.

The InstructGPT paper primarily used binary choices with a third "tie" option.


### 2.4 Annotation Guidelines: The Hidden Art

Writing annotation guidelines is one of the most underestimated tasks in RLHF. These guidelines tell annotators what "better" means. Bad guidelines = inconsistent data = broken reward model.

Good guidelines include:

1. **Concrete examples** of good vs bad responses for each criterion
2. **Priority ordering** when criteria conflict (e.g., helpfulness vs safety)
3. **Edge case handling** — what to do with ties, refusals, or unsafe content
4. **Regular calibration sessions** to keep annotators aligned

The OpenAI InstructGPT guidelines reportedly ran to dozens of pages. Anthropic published a simplified version of their guidelines in the Constitutional AI paper. The key insight: **guidelines are a living document** that evolves as you discover ambiguity.

### 2.5 Who Are the Annotators?

This is a crucial practical decision:

| Type | Quality | Speed | Cost/Comparison | Best For |
|------|---------|-------|-----------------|----------|
| Expert researchers | Very high | Slow (~5/min) | $5–$20 | Gold standard, small datasets |
| Trained contractors | Good | Medium (~15/min) | $0.50–$2 | Production RLHF |
| Domain specialists | High (in domain) | Medium | $1–$5 | Specialized models |
| Crowd workers | Variable | Fast (~30/min) | $0.02–$0.10 | Large-scale, simple tasks |

![Figure 3: Annotation Quality Metrics](../zh/images/day14/annotation-quality-metrics-v2.png)
*Caption: Left: The distribution of inter-annotator agreement scores. Right: The quality-speed trade-off for different annotator types*

Most production RLHF pipelines use a **tiered approach**: expert researchers create the guidelines and validate quality, trained contractors do the bulk of the annotation, and crowd workers handle simple, well-defined tasks.

### 2.6 The Comparison Process

Now we get to the heart of preference annotation: **how annotators actually compare responses**. The design of this process directly impacts data quality.

**Blind comparison** means annotators don't know which model generated which response. This is crucial because knowing the source introduces bias — an annotator might favor responses from a "known good" model or discount responses they think came from a weaker one. Some teams go further and **shuffle the presentation order** of responses (randomly assigning which appears as "Response A" vs "Response B") to prevent **position bias**, where annotators unconsciously prefer whichever they see first.

> **Fun fact**: Position bias is well-documented in survey research. In election ballots, candidates listed first get a measurable boost in votes. The same effect applies to response comparisons — all else being equal, "Response A" gets picked slightly more often than "Response B" just by being shown first. Shuffling neutralizes this.

**Multiple annotators per pair** (typically 2–3) are essential for reliability. A single annotator might be having a bad day, misunderstanding of guidelines, or simply making an error. By having overlapping annotators judge the same pairs, you can:
- Measure agreement (via Cohen's Kappa, which we'll cover below)
- Identify consistently poor annotators
- Catch genuinely ambiguous pairs where reasonable people disagree

**Disagreement resolution** is what happens when annotators don't agree. There are several strategies:
- **Majority vote**: The most common approach. With 3 annotators, majority wins. Simple and effective.
- **Escalation**: Disputed pairs are sent to a senior annotator or expert for a tie-breaking decision.
- **Discard**: Remove pairs where annotators disagree significantly. This reduces dataset size but increases average quality.
- **Keep as disagreement signal**: Controversial pairs — where competent annotators genuinely disagree — are actually *informative*. They tell you that responses are close in quality, which is useful signal for the reward model. Some teams explicitly mark these as "tie" or "near-tie" rather than discarding them.

> **Key insight**: Perfect annotator agreement is neither realistic nor desirable. If every annotator always agrees, your task is probably too easy and you're not collecting nuanced signal. Some level of disagreement (κ ≈ 0.5–0.7) is the sweet spot — consistent enough to be reliable, but with enough variation to capture genuine subjectivity.

### 2.7 Data Cleaning and Balancing

Before the preference dataset goes into reward model training, it needs quality checks and balancing.

**Toxicity filtering & safety checks** are non-negotiable. Scan responses for harmful content before including them in the dataset. Why? If your dataset contains toxic "chosen" responses — responses that are rude, biased, or harmful — your reward model will learn to prefer toxicity. Automated toxicity classifiers help catch obvious cases, but manual review is also needed for subtle issues.

**Balanced dataset for RLHF** means equal representation across topics, difficulty levels, and response types. This is about coverage, not just random sampling. If 80% of your preference data is about coding questions and 20% about creative writing, your reward model will be heavily biased toward coding quality and may degrade creative performance. Stratified sampling ensures proportional coverage across all dimensions.

Think of it like training a student — if you only give them math problems, they'll struggle with literature. A balanced curriculum produces a well-rounded student.

---

## 3. Measuring Annotation Quality

How do you know if your preference data is any good? This is where quality metrics come in.

### 3.1 Inter-Annotator Agreement

The most important metric is **inter-annotator agreement** — do different annotators reach the same conclusion when shown the same pair of responses?

The standard measure is **Cohen's Kappa (κ)**:

$$
\begin{aligned}
\kappa &= \frac{p_o - p_e}{1 - p_e}
\end{aligned}
$$

Where $p_o$ is the observed agreement rate and $p_e$ is the expected agreement by chance.

**Interpretation:**

| Kappa | Agreement Level |
|-------|----------------|
| < 0.20 | Poor |
| 0.20 – 0.40 | Fair |
| 0.40 – 0.60 | Moderate |
| 0.60 – 0.80 | Good |
| > 0.80 | Very Good |

For RLHF, you typically want κ > 0.6 at minimum. The InstructGPT paper reported agreement rates around 73% (which translates to roughly κ ≈ 0.5–0.6 depending on the task).

When agreement is low, it usually means one of:
- **Ambiguous guidelines** — annotators interpret "better" differently
- **Genuinely subjective comparisons** — both responses are equally good
- **Poor annotator training** — they don't understand the task

### 3.2 Gold Standard Sets

Most teams maintain a **gold standard** set of ~500–1000 comparisons where the "correct" answer is known (agreed upon by experts). Every batch of annotations is checked against this gold set.

Annotators who consistently disagree with the gold standard are either retrained or removed. This is a standard quality control technique from the survey research literature.

### 3.3 Adversarial Testing

A clever technique: deliberately insert **obviously wrong** responses into the annotation pipeline. If annotators consistently fail to reject these, you have a quality problem. This is called "attention checking" in the survey literature and "adversarial quality control" in ML.

---

## 4. The Economics of RLHF

Let's talk money. How much does RLHF actually cost?

![Figure 4: RLHF Cost Breakdown](../zh/images/day14/rlhf-cost-breakdown.png)
*Caption: Left: Where the money goes in a typical RLHF project. Right: How costs scale with project ambition*

### 4.1 Cost Breakdown

A typical RLHF project for a medium-sized model (~7B–70B parameters) costs roughly:

| Item | Cost | Percentage |
|------|------|------------|
| Annotation labor | $50K–$500K | ~45% |
| Compute infrastructure | $25K–$250K | ~25% |
| Quality control | $15K–$150K | ~15% |
| Management & training | $10K–$100K | ~10% |
| Iteration & redo | $5K–$50K | ~5% |

**Key insight**: Labor dominates. The compute costs of RLHF training are actually modest compared to pre-training. The bottleneck is the human annotation.

### 4.2 Scaling Laws for RLHF Data

How much preference data do you actually need? The answer depends on what you're optimizing:

- **Helpfulness**: 100K–500K comparisons for significant improvement
- **Safety/harmlessness**: 50K–200K comparisons (fewer edge cases)
- **Specialized domains**: 10K–50K comparisons (but needs domain experts)

The relationship between data volume and reward model quality follows roughly a **power law**:

$$
\begin{aligned}
\text{Reward Model Accuracy} &\propto N^{\alpha}
\end{aligned}
$$

Where $N$ is the number of preference comparisons and $\alpha \approx 0.3$–$0.5$ in practice. This means **doubling your data gives you diminishing returns** — the first 100K comparisons are worth more than the next 100K.

### 4.3 How Preference Data Becomes a Reward Model

You might wonder: we have pairwise comparisons ("A is better than B"), but the reward model needs to output a **scalar score** for any response. How do we bridge this gap?

The standard approach uses the **Bradley-Terry model**, originally developed for ranking chess players. The key idea: each response has a latent "reward score" $r(x, y)$, and the probability of preferring response $y_1$ over $y_2$ given prompt $x$ is:

$$
\begin{aligned}
P(y_1 \succ y_2 \mid x) &= \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))} = \sigma(r(x, y_1) - r(x, y_2))
\end{aligned}
$$

Where $\sigma$ is the sigmoid function. This is elegant: **the reward model only needs to learn relative differences, not absolute scores**. The loss function for training the reward model is simply the negative log-likelihood of the observed preferences:

$$
\begin{aligned}
\mathcal{L} &= -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]
\end{aligned}
$$

Where $y_w$ is the winning (chosen) response and $y_l$ is the losing (rejected) response. This is why the **quality of pairwise comparisons matters so much** — the reward model directly learns from these judgments. Noisy labels → noisy reward signal → reward hacking during PPO.

### 4.4 Open Source RLHF Datasets

You don't have to collect everything from scratch. Several high-quality open datasets exist:

1. **Open Assistant Conversations** (LAION): ~161K conversation trees with human rankings — [https://huggingface.co/datasets/OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)
2. **HH-RLHF** (Anthropic): ~170K helpfulness and harmlessness comparisons — [https://huggingface.co/datasets/Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
3. **UltraFeedback** (Argilla): ~64K comparisons using GPT-4 as judge — [https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)
4. **SHP** (Stanford): ~385K comparisons from Reddit upvotes — [https://huggingface.co/datasets/stanfordnlp/SHP](https://huggingface.co/datasets/stanfordnlp/SHP)

These datasets are great starting points, but they have limitations: they may not reflect your specific use case, and the quality varies. Most production systems start with open data and then supplement with custom collection.

---

## 5. RLAIF: AI-as-Annotator

The most exciting recent development is using AI models as annotators instead of (or in addition to) humans. This approach is called **RLAIF (Reinforcement Learning from AI Feedback)**.

![Figure 5: Traditional RLHF vs RLAIF / Constitutional AI](../zh/images/day14/rlhf-vs-rlaif-v2.png)
*Caption: Traditional RLHF relies on human annotators; RLAIF replaces step 3 with AI evaluation, reducing cost by 10-100x*

### 5.1 Constitutional AI

Anthropic's Constitutional AI (CAI) approach replaces human preference annotation with AI self-evaluation:

1. The model generates multiple responses
2. An AI (typically a stronger model like Claude or GPT-4) evaluates each response against a set of principles (the "constitution")
3. The AI's preferences are used to train the reward model

The "constitution" is a list of rules like:
- "Choose the response that is most helpful while being harmless"
- "Choose the response that is most honest and least deceptive"
- "Choose the response that is least offensive and controversial"

### 5.2 Does It Work?

Surprisingly well. Several papers have shown that RLAIF produces models that are competitive with human-annotated RLHF:

- **Cost reduction**: 10x–100x cheaper
- **Speed**: Days instead of months
- **Scalability**: Virtually unlimited data
- **Consistency**: AI annotators don't have bad days

But there are risks:
- **AI bias amplification**: The AI annotator's biases get baked in
- **Reward hacking**: Models learn to game the AI annotator, not to be genuinely better
- **Homogeneity**: AI preferences may lack the diversity of human preferences

### 5.3 The Hybrid Approach

Most practitioners now use a **hybrid** approach:
1. Start with RLAIF for the bulk of data (cheap, fast)
2. Use human annotators for quality control and edge cases
3. Continuously validate AI annotations against human judgments

This gets you 80% of the quality at 10% of the cost.

---

## 6. Code Example: Building a Preference Dataset

Here's a practical example of how to create a preference dataset:

```python
import json
import random
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PreferenceExample:
    """A single preference comparison example."""
    prompt: str
    chosen_response: str
    rejected_response: str
    annotator_id: str
    confidence: float  # 0.0-1.0

def collect_preference_data(
    prompts: List[str],
    model_generate,  # function that generates responses
    num_responses_per_prompt: int = 4,
    min_confidence: float = 0.7,
) -> List[PreferenceExample]:
    """Collect preference comparisons for a set of prompts."""
    
    dataset = []
    
    for prompt in prompts:
        # Step 1: Generate multiple diverse responses
        responses = []
        for i in range(num_responses_per_prompt):
            # Vary temperature for diversity
            temp = 0.7 + (i * 0.1)
            resp = model_generate(prompt, temperature=temp)
            responses.append(resp)
        
        # Step 2: Get pairwise comparisons
        # (In practice, this goes to human annotators)
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                comparison = get_human_comparison(
                    prompt, responses[i], responses[j]
                )
                
                if comparison.confidence >= min_confidence:
                    dataset.append(PreferenceExample(
                        prompt=prompt,
                        chosen_response=comparison.winner,
                        rejected_response=comparison.loser,
                        annotator_id=comparison.annotator_id,
                        confidence=comparison.confidence,
                    ))
    
    return dataset

def compute_cohens_kappa(
    annotations_a: List[int],
    annotations_b: List[int],
) -> float:
    """Compute Cohen's Kappa for two annotators."""
    n = len(annotations_a)
    assert n == len(annotations_b)
    
    # Observed agreement
    p_o = sum(a == b for a, b in zip(annotations_a, annotations_b)) / n
    
    # Expected agreement by chance
    categories = set(annotations_a + annotations_b)
    p_e = 0.0
    for c in categories:
        p_a = sum(a == c for a in annotations_a) / n
        p_b = sum(b == c for b in annotations_b) / n
        p_e += p_a * p_b
    
    if p_e == 1.0:
        return 1.0
    
    return (p_o - p_e) / (1 - p_e)

# Example usage
if __name__ == "__main__":
    # Simulate annotation quality check
    ann_a = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1]  # Annotator A's labels
    ann_b = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]  # Annotator B's labels
    
    kappa = compute_cohens_kappa(ann_a, ann_b)
    print(f"Cohen's Kappa: {kappa:.3f}")
    # Interpretation: > 0.6 is good, > 0.8 is very good
```

---

## 7. Common Misconceptions

### ❌ "More data is always better"

Not true for RLHF. Bad annotations actively hurt your reward model. It's better to have 50K high-quality comparisons than 500K noisy ones. The signal-to-noise ratio matters more than raw volume.

### ❌ "GPT-4 can replace all human annotators"

RLAIF is powerful, but purely AI-generated preferences can create echo chambers. The model optimizes for what the AI annotator likes, which may not align with what humans actually want. Human oversight remains essential.

### ❌ "Preference annotation is just asking 'which is better'"

The annotation guidelines, training, and quality control infrastructure are where the real complexity lives. Two teams can use the same model and same annotation platform, and get wildly different results based solely on how they run their annotation pipeline.

---

## 8. Further Reading

### Beginner
1. [InstructGPT Paper](https://arxiv.org/abs/2203.02155) — The original RLHF paper with detailed annotation methodology
2. [Open Assistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst2) — Explore a real preference dataset on Hugging Face

### Advanced
1. [Constitutional AI](https://arxiv.org/abs/2212.08073) — Anthropic's approach to AI-generated feedback
2. [RLAIF Paper](https://arxiv.org/abs/2309.00267) — Google DeepMind's systematic comparison of RLAIF vs RLHF

### Tools
1. [Argilla](https://argilla.io/) — Open-source annotation platform designed for LLM feedback
2. [LMSYS Chatbot Arena](https://chat.lmsys.org/) — Large-scale preference collection in action

---

## Reflection Questions

1. If annotation quality is more important than quantity, how would you design a system that maximizes quality per dollar spent?
2. What are the ethical implications of outsourcing preference annotation to workers in lower-cost countries? How would you address fairness concerns?
3. Can a model trained purely on AI feedback ever be truly "aligned" with human values? What's missing?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| SFT Data | Human-written demonstrations teaching the model *how* to respond (10K-50K examples) |
| Preference Data | Human rankings of model outputs, used to train the reward model (100K-1M comparisons) |
| Prompt Data | Diverse prompts for RL training, no responses needed (50K-500K prompts) |
| Cohen's Kappa | Metric for measuring inter-annotator agreement; > 0.6 is acceptable |
| RLAIF | Using AI models as annotators instead of humans; 10-100x cheaper |
| Constitutional AI | Anthropic's approach: AI evaluates responses against a set of principles |
| Hybrid Approach | Combining cheap RLAIF bulk data with human quality control |

**Key Takeaway**: The data pipeline — not the algorithm — is the bottleneck in RLHF. Collecting high-quality preference comparisons requires careful annotation guidelines, robust quality control, and significant investment. RLAIF is making this dramatically cheaper, but human oversight remains essential for alignment.

---

*Day 14 of 60 | LLM Fundamentals*
*Word count: ~2700 | Reading time: ~14 minutes*
