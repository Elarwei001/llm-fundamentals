# Day 15: DPO and Alternatives — Why People Are Abandoning PPO

> **Core Question**: If RLHF with PPO is the alignment method behind ChatGPT, why is the research community rapidly moving to simpler alternatives like DPO — and which method should you use?

---

## Opening

Imagine you're a teacher grading essays. With PPO-based RLHF, you'd need to: (1) train a separate "grader" AI to score essays, (2) use that grader as a reward signal, (3) carefully tune a reinforcement learning algorithm so the student AI improves without collapsing. It works — ChatGPT proved it — but it's like requiring three separate classrooms, two teachers, and a carefully controlled environment just to give feedback on homework.

What if instead, you could just show the student "this essay is better than that one" and let them figure out the rest? That's the promise of DPO (Direct Preference Optimization) and its growing family of alternatives. Since 2023, the field has seen an explosion of methods that skip the reward model entirely, and the results are remarkable: simpler code, faster training, and often better outputs.

Today we'll understand why PPO is falling out of favor, how DPO works under the hood, and what the landscape looks like in 2024-2025.

---

## 1. The Problem with PPO-Based RLHF

### 1.1 A Quick Recap of RLHF

As we covered in Days 13-14, the standard RLHF pipeline has three stages:

1. **Supervised Fine-Tuning (SFT)**: Train the model on good demonstrations
2. **Reward Model (RM) Training**: Train a separate model to score responses
3. **PPO (Proximal Policy Optimization)**: Use the reward model to fine-tune the policy via RL

![Figure 1: RLHF vs DPO Pipeline Comparison](../zh/images/day15/rlhf-vs-dpo-pipeline.png)
*Caption: The three-stage RLHF pipeline (left) vs. DPO's single-stage approach (right). DPO eliminates the reward model entirely.*

### 1.2 Why PPO is Painful

PPO works, but it's brutal in practice. Here are the concrete issues:

**Four models in memory simultaneously.** During PPO training, you need: (1) the policy model being trained, (2) a reference model (frozen copy for KL penalty), (3) the reward model, and (4) often a value head (critic). For a 7B parameter model, that's ~28B parameters loaded at once. For a 70B model? Good luck.

**Reward hacking.** The policy learns to exploit gaps in the reward model rather than genuinely improving. You might get responses that score high on the reward model but are actually worse — verbose, repetitive, or sycophantic.

**Hyperparameter sensitivity.** PPO has a notoriously large hyperparameter search space: learning rate, clip range, KL coefficient, GAE lambda, value function coefficient, discount factor... A bad configuration can cause training to diverge silently.

**Training instability.** Unlike supervised learning where loss monotonically decreases, PPO loss can spike unpredictably. Researchers report needing multiple restarts and careful monitoring.

```python
# PPO requires managing multiple models - this is the complexity
import torch.nn as nn

class PPOTrainingSetup:
    """You need ALL of these running simultaneously."""
    def __init__(self, base_model):
        self.policy = base_model                  # Being optimized
        self.ref_policy = copy.deepcopy(base_model)  # Frozen reference
        self.reward_model = RewardModel(base_model.config)  # Separate model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)  # Critic
        
        # Memory: ~4x the base model size
        total_params = sum(p.numel() for p in self.policy.parameters())
        print(f"Base model: {total_params/1e9:.1f}B params")
        print(f"Total in memory: ~{4 * total_params/1e9:.1f}B params")
```

---

## 2. DPO: Direct Preference Optimization

### 2.1 The Key Insight

DPO's breakthrough (Rafailov et al., NeurIPS 2023) is a mathematical trick: **you can skip the reward model because the optimal policy under the RLHF objective has a closed-form solution**.

> **What is a closed-form solution?** A "closed-form solution" means you can write down the exact answer using a formula — no iteration, no trial-and-error, no gradual approximation needed. For example, the quadratic formula $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$ is a closed-form solution: plug in the coefficients, get the answer immediately. In contrast, most RL problems (including PPO) have *no* closed-form solution — you can only approximate the answer through repeated trial-and-error updates. DPO's key insight is that, under the KL-regularized RLHF objective, the optimal policy *does* have one, which means we can solve it directly instead of iterating.

Here's the intuition. In RLHF, we train a reward model $r(x, y)$, then optimize:

$$
\begin{aligned}
\max_{\pi} \; \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta \cdot D_{KL}(\pi \| \pi_{ref})
\end{aligned}
$$

*Formula 1: The KL-regularized RLHF objective.*

> **What is this formula doing?** This is the RLHF objective — two forces pulling in opposite directions:
> - **First term** $\mathbb{E}[r(x,y)]$: maximize reward — make the model generate high-scoring responses
> - **Second term** $\beta \cdot D_{KL}$: stay close to the original SFT model — don't drift too far
> - **$\beta$** is the knob balancing these two forces. High β → conservative (stay close to SFT); Low β → aggressive (chase high rewards)

The optimal solution to this is:

$$
\begin{aligned}
\pi^{*}(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)
\end{aligned}
$$

*Formula 2: The closed-form solution — optimal policy in terms of reward.*

> **What is this formula doing?** This is the *exact answer* to Formula 1 — the closed-form solution. It says: **optimal policy = reference model × exp(reward/β)**.
> - $\pi^*(y|x)$ (with the asterisk) = the **optimal policy** — the best possible model we're trying to find
> - $\pi_{ref}(y|x)$ (with "ref" subscript) = the **reference model** — the frozen SFT model, our starting point
> - The single $\exp\left(\frac{1}{\beta} r(x,y)\right)$ acts as a multiplier: high reward → larger multiplier → higher probability for that response
> - $Z(x)$ is just a normalizing constant (ensures probabilities sum to 1) — it's not important, and will cancel out later

where $Z(x)$ is a normalizing constant.

> **How do we get from Formula 2 to Formula 3?** Just two steps — take log, then rearrange:
>
> **Step 1:** Take $\log$ of both sides of Formula 2:
>
> $\log \pi^*(y|x) = -\log Z(x) + \log \pi_{ref}(y|x) + \frac{1}{\beta} r(x,y)$
>
> **Step 2:** Move $\log \pi_{ref}(y|x)$ to the left, multiply both sides by $\beta$:
>
> $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$
>
> That's Formula 3 — just Formula 2 rearranged. Nothing new was introduced.

Rearranging, we can express the **implicit reward** in terms of the policy:

$$
\begin{aligned}
r(x,y) = \beta \log \frac{\pi^{*}(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
\end{aligned}
$$

*Formula 3: Implicit reward — reward expressed in terms of the policy (derived from Formula 2).*

> **What is this formula doing?** This is Formula 2 rearranged — now we're *backing out* the reward from the policy instead of computing the policy from the reward.
> - **Key insight**: when we compare two responses $y_w$ (good) and $y_l$ (bad), we write Formula 3 for *each* of them:
>   - $r(x, y_w) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} + \beta \log Z(x)$
>   - $r(x, y_l) = \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} + \beta \log Z(x)$
>   - $Z(x)$ depends only on the prompt $x$, NOT on the response $y$ — so it's the same in both equations
>   - Subtracting: $\beta \log Z(x) - \beta \log Z(x) = 0$ — **it cancels!**
> - So we never need to know the actual reward value — only the *relative difference* between two responses matters
> - This is exactly why we can skip the reward model entirely

The key: $Z(x)$ cancels out when we compare two responses! So we can directly optimize the policy using preference pairs without ever training a reward model.

But how exactly do we turn this into a training objective? We need one more ingredient: the **Bradley-Terry model**. This model assumes that human preference probability follows a sigmoid of the reward difference:

$$
\begin{aligned}
P(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)
\end{aligned}
$$

*Formula 3.5: The Bradley-Terry preference model — "the probability that humans prefer $y_w$ over $y_l$ is the sigmoid of the reward gap."*

Now substitute Formula 3's expression for $r(x,y_w) - r(x,y_l)$ (after $Z(x)$ cancels) into this, and you get Formula 4 — the DPO loss.

### 2.2 The DPO Loss Function

Given a preference pair $(y_w, y_l)$ where $y_w$ is chosen (winner) and $y_l$ is rejected (loser):

$$
\begin{aligned}
\mathcal{L}_{DPO} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
\end{aligned}
$$

*Formula 4: The DPO loss function — the only formula you actually need to implement.*

where $\sigma$ is the sigmoid function and $\beta$ controls how much the policy deviates from the reference.

> **What is this formula doing?** It looks intimidating, but let's break it down:
> - $\log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)}$ = how much the policy shifted its probability on the **good** response
> - $\log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}$ = how much the policy shifted its probability on the **bad** response
> - **Subtracting the two** = the gap between how much the good response went up vs. the bad response went up
> - The outer $\log \sigma()$ turns this gap into a 0–1 probability, and the negative log penalizes when the gap is too small
>
> **In plain English:** make the model increase probability on the good response MORE than on the bad response. The bigger the gap, the lower the loss. It's the same logic as training a classifier — except here we're classifying "good response vs. bad response" instead of "cat vs. dog."

![Figure 2: DPO Loss Function Visualization](../zh/images/day15/dpo-loss-visualization.png)
*Caption: Left: The DPO loss landscape — when the model already prefers the chosen response (positive margin), loss is low. Right: The beta parameter controls how strictly the model follows preferences.*

> **How to read these charts:**
> - **Horizontal axis (Log-Ratio Margin):** This is the value inside the sigmoid in Formula 4 — how much more the model prefers the good response over the bad one, relative to the reference model.
>   - **> 0 (right):** Model already prefers the good response → loss is low (doing well!)
>   - **= 0 (center):** Model treats both equally → loss ≈ 0.69 (random guessing, like a coin flip)
>   - **< 0 (left):** Model actually prefers the bad response → loss is high (needs correction)
> - **Left chart:** Shows the overall shape of the DPO loss — it's just $-\log \sigma(x)$, decreasing as the margin grows.
> - **Right chart:** Shows how $\beta$ controls steepness. High β → steep → aggressive updates; Low β → flat → gentle updates.

Think of it like this: DPO is a **race between two runners**. Instead of giving each runner an absolute score (like a reward model), you just measure the gap between them and push the gap wider. The loss is low when the policy assigns much more probability to $y_w$ than $y_l$ relative to the reference model.

### 2.3 What Beta Actually Does

The $\beta$ parameter (sometimes called temperature) is the most important hyperparameter in DPO:

- **Low beta (0.05-0.1)**: The policy changes slowly. More stable, but may underfit preferences. Good for subtle style alignment.
- **High beta (0.5-1.0)**: The policy aggressively follows preferences. Risk of overfitting or losing capabilities. Use when preferences are strong and reliable.

A practical rule of thumb: start with $\beta = 0.1$ and increase if the model isn't changing enough, decrease if it's losing capabilities.

### 2.4 DPO in Code

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Compute DPO loss.
    
    Args:
        policy_chosen_logps: log pi(y_w|x) under current policy
        policy_rejected_logps: log pi(y_l|x) under current policy
        ref_chosen_logps: log pi_ref(y_w|x) under reference
        ref_rejected_logps: log pi_ref(y_l|x) under reference
        beta: temperature controlling deviation from reference
    """
    # Log-ratio: how much has the policy shifted from reference?
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    
    # The margin: how much more does policy prefer chosen vs rejected?
    logits = beta * (chosen_logratios - rejected_logratios)
    
    # Logistic loss (same as binary cross-entropy with margin)
    loss = -F.logsigmoid(logits).mean()
    
    return loss

# Key insight: only need 2 forward passes (chosen + rejected)
# Compare with PPO: 4 models × multiple rollouts × reward scoring
```

---

## 3. Training Dynamics and Practical Considerations

### 3.1 Why DPO is More Stable

DPO converges to a fixed objective (like supervised learning), while PPO is an iterative RL process. This fundamental difference means:

| Property | PPO | DPO |
|----------|-----|-----|
| Objective | Changes as policy shifts | Fixed dataset, fixed loss |
| Number of models | 4 (policy, ref, reward, critic) | 2 (policy, ref) |
| Memory | ~4× base model | ~2× base model |
| Hyperparameters | 7-10 critical ones | 1-2 (beta, learning rate) |
| Training curves | Noisy, may diverge | Smooth, monotonic |
| Reward hacking | Common | Impossible (no reward model) |

![Figure 3: DPO Training Dynamics](../zh/images/day15/dpo-training-dynamics.png)
*Caption: Left: DPO training is smooth and stable compared to PPO's noisy dynamics. Right: The implicit reward margin (chosen - rejected) steadily increases during DPO training.*

> **How to read these charts:**
> - **Left chart:** Compares training loss curves. DPO (blue) decreases smoothly; PPO (orange) is noisy with spikes. Note: DPO isn't perfectly immune to instability — with noisy preference data or excessively high β, loss can still oscillate. But it's far more predictable than PPO.
> - **Right chart:** The blue line shows the implicit reward margin ($r_{\text{implicit}}(y_w) - r_{\text{implicit}}(y_l)$) over training steps. As training progresses, this margin grows — meaning the model is learning to prefer chosen responses more and more. The light blue band/shading likely represents variance or a confidence interval across training runs.

### 3.2 The Reference Model Problem

DPO still needs a frozen reference model ($\pi_{ref}$), which doubles memory. This is because DPO needs to know "how much has the policy changed" — without the reference, there's no baseline.

Some newer methods address this:

### 3.2.1 SimPO: Simple Preference Optimization

**What it removes:** The reference model entirely. SimPO (Meng et al., 2024) doesn't need $\pi_{ref}$ at all.

**How it works:** In DPO, the reference model serves as a baseline to measure "how much has the policy changed." SimPO replaces this with a clever trick: use the **model's own average log-probability** of the response as the baseline.

Instead of computing $\log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$, SimPO replaces the log-ratio with a **length-normalized log-probability** — no reference model needed:

$$\tilde{r}(x,y) = \frac{\beta}{|y|} \log \pi(y|x) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi(y_t|x, y_{<t})$$

*Formula 5: SimPO uses length-normalized log-probability as the implicit reward, replacing the $\log \frac{\pi}{\pi_{ref}}$ ratio entirely.*

> **What does this mean?** Instead of asking "how much did the policy change vs. the frozen model?", SimPO asks "what is the average per-token log-probability of this response?" The length normalization ($\frac{1}{|y|}$) ensures longer responses aren't unfairly favored (since longer sequences have lower total log-prob by definition). The chosen response should have higher average log-prob than the rejected one.

**Why it works:** The intuition is that a well-aligned model should assign *consistently high* probability to good responses (not just high probability on a few tokens). By comparing against the response's own average, SimPO achieves implicit normalization without needing a separate reference model.

**Memory savings:** From ~2× base model (DPO) down to ~1× — only the policy model is needed.

### 3.2.2 ORPO: Odds Ratio Preference Optimization

**What it removes:** The reference model AND the separate alignment stage. ORPO (Hong et al., 2024) does SFT + alignment in a single pass.

**How it works:** During normal SFT training on chosen responses, ORPO adds an extra loss term that penalizes the model when it can't distinguish chosen from rejected:

> **Why can't SFT distinguish good from bad?** Standard SFT only trains on good responses — it learns to generate them, but has *never seen bad responses*, so it doesn't know what "bad" looks like. A model trained only with SFT might assign high probability to both good and bad responses equally, because nobody told it that the bad ones are bad. ORPO fixes this by feeding it both chosen and rejected responses during the same training step: the SFT loss teaches "generate good stuff," and the odds-ratio loss teaches "good and bad are different — prefer good."

$$
\begin{aligned}
\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}
\end{aligned}
$$

*Formula 6: ORPO combines SFT loss with an odds-ratio alignment loss.*

> **What is the odds ratio?** The odds of generating the chosen response are $\frac{\pi(y_w|x)}{1 - \pi(y_w|x)}$. Similarly for rejected. The odds ratio is:
>
> $$\text{OR} = \frac{\pi(y_w|x) / (1 - \pi(y_w|x))}{\pi(y_l|x) / (1 - \pi(y_l|x))}$$
>
> If OR ≈ 1, the model treats chosen and rejected equally (bad). If OR >> 1, the model correctly prefers chosen (good). $\mathcal{L}_{OR}$ pushes OR away from 1.

**Why it works:** Standard SFT only sees good responses — it has no signal about what to avoid. ORPO fixes this by feeding it both chosen and rejected responses during the same training step. The SFT loss teaches it to generate good responses; the odds-ratio loss teaches it to *distinguish* good from bad.

**Memory savings:** ~1× base model — only one model, one training stage. The simplest possible pipeline.

### 3.3 Common Pitfalls

**Edit distance matters.** DPO works best when $y_w$ and $y_l$ are similar in structure but differ in quality. If they're completely different, the model may learn spurious features rather than the actual preference.

> **What does "similar structure, different quality" mean?** Here's a concrete example. Suppose the prompt is "Explain what a black hole is":
>
> **Good preference pair (similar structure, different quality):**
> - *Chosen:* "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. It forms when a massive star collapses. The event horizon marks its boundary."
> - *Rejected:* "A black hole is a very dark thing with strong gravity, stuff goes in and can't come out. Stars explode and become black holes."
> - Both have the same 3-part structure (what it is → how it forms → key concept), but the chosen version is precise and complete while the rejected version is vague and colloquial.
>
> **Bad preference pair (completely different structure):**
> - *Chosen:* "A black hole is a region of spacetime where gravity is so strong..." (precise 3-sentence explanation)
> - *Rejected:* "Haha black holes are so cool, I saw a movie about them! They're just really powerful stuff I guess."
> - These are too different — the model might learn "formal vs. casual tone" instead of "accurate vs. inaccurate content."
>
> **Why structure matters:** DPO only adjusts the ranking of the existing probability distribution. When responses are structurally similar, the model only needs to fine-tune "which words are better." When they're completely different, it may pick up on surface-level style differences instead of substantive quality differences.

**Capacity for editing, not generating.** DPO primarily teaches the model to reorder its existing probability distribution, not to generate genuinely new capabilities. If you need the model to learn new skills, SFT first, then DPO.

**Overfitting to preferences.** With small datasets (< 5K pairs), DPO can memorize specific patterns. Regular evaluation on held-out tasks is essential.

---

## 4. The Broader Landscape: DPO Alternatives

### 4.1 IPO: Identity Preference Optimization

IPO (Azar et al., 2023) addresses a theoretical weakness of DPO: it assumes the preference data comes from a Bradley-Terry model (a specific preference model). IPO instead directly optimizes a regret bound:

> **What is regret?** In decision theory, regret = the gap between "what my strategy achieved" and "what the optimal strategy would have achieved." It's not about how much you lost — it's about how much *more* you could have won. For example, if the best possible policy scores 100 and yours scores 70, your regret is 30.
>
> **What is a regret bound?** A mathematical guarantee that says "no matter what, the gap between my algorithm and the optimal strategy won't exceed X." It gives you a worst-case performance floor.
>
> **Why does IPO use regret instead of Bradley-Terry?** DPO assumes human preferences follow the Bradley-Terry model ($P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$). If this assumption is wrong (it often is — human annotations are noisy and inconsistent), DPO's guarantees break down. IPO avoids this assumption entirely by directly minimizing the worst-case regret.

$$
\begin{aligned}
\mathcal{L}_{IPO} = \mathbb{E}\left[ \left( \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]
\end{aligned}
$$

Key difference: IPO uses a **squared loss** instead of logistic loss, which makes it more robust when preferences are noisy (which they almost always are in practice).

> **How the regret bound shows up in the formula:**
> - The outer **squared** $(\cdot)^2$ replaces DPO's $\log \sigma(\cdot)$ — squared loss is more robust to outliers (a single bad annotation can't blow up the loss)
> - The **$-\frac{1}{2\beta}$** term comes directly from the regret bound derivation — it ensures we're minimizing actual regret, not just fitting a possibly-wrong preference model
> - Together, these changes mean IPO provides guarantees even when the Bradley-Terry assumption is violated

### 4.2 KTO: Kahneman-Tversky Optimization

KTO (Ethayarajh et al., 2023) is a breakthrough for data-constrained settings.

> **Background: Why was KTO needed?**
>
> All the methods above (DPO, IPO) require **paired preference data**: for each prompt, you need two responses ($y_w$ and $y_l$) and a human judgment of which is better. This is expensive and limiting:
> - You must generate multiple responses per prompt
> - Humans must carefully compare and rank them
> - If you already have a dataset of single responses with quality ratings (e.g., "thumbs up" / "thumbs down"), you can't use DPO at all
>
> In practice, many real-world datasets are **unpaired**: customer feedback (helpful / not helpful), moderation labels (safe / unsafe), quality scores (good / bad). KTO was designed to work with exactly this kind of data.

**Inspired by prospect theory from behavioral economics:**

> **What is prospect theory?** Developed by Daniel Kahneman and Amos Tversky (Kahneman won the 2002 Nobel Prize in Economics for this work). The key finding: **humans are not rational** when evaluating gains and losses. Specifically:
> - **Loss aversion**: losing $100 feels roughly twice as painful as gaining $100 feels good
> - **Diminishing sensitivity**: the difference between $0 and $100 matters more than the difference between $1000 and $1100
>
> **How does this connect to alignment?** When training a language model, "desirable" responses are gains and "undesirable" responses are losses. KTO applies the same asymmetry: the model should penalize bad responses *more strongly* than it rewards good ones. This matches human intuition — you'd rather your assistant occasionally give a mediocre response than ever give a truly harmful one.

- **Does NOT require paired preference data** — only binary labels (good/bad)
- This is huge because paired data is expensive: you need to generate two responses and have humans rank them
- KTO only needs "this response is good" or "this response is bad" — like a simple thumbs up / thumbs down

The loss weights positive and negative examples differently, reflecting how humans are loss-averse (bad outcomes hurt more than good outcomes please):

```python
def kto_loss(policy_logps, ref_logps, labels, beta=0.1):
    """
    KTO: only needs binary labels, not preference pairs.
    labels: 1 for desirable, 0 for undesirable
    """
    logratios = policy_logps - ref_logps
    
    # Desirable responses: push policy above reference
    # Undesirable responses: push policy below reference
    des_mask = (labels == 1)
    und_mask = (labels == 0)
    
    loss_des = -F.logsigmoid(beta * logratios[des_mask]).mean()
    loss_und = -F.logsigmoid(-beta * logratios[und_mask]).mean()
    
    # Weight by proportion (balanced loss)
    total = des_mask.sum() + und_mask.sum()
    w_des = und_mask.sum() / total
    w_und = des_mask.sum() / total
    
    return w_des * loss_des + w_und * loss_und
```

> **FAQ: What if my labels aren't binary? Can KTO handle multi-level ratings (bad, OK, good, excellent)?**
>
> KTO's loss is designed for binary labels — push good responses up, push bad responses down. Multi-level labels would require redesigning the loss (e.g., different weights or margins per level), which would make it a new algorithm entirely.
>
> **What to do instead if you have multi-level labels:**
> - **Best option:** Convert to binary (bad/OK → undesirable, good/excellent → desirable) and use KTO. This is what most practitioners do, because annotation noise makes fine-grained distinctions unreliable anyway.
> - **Alternative:** Pair adjacent levels as preference pairs (excellent vs good, good vs OK, OK vs bad) and use DPO.
> - **Key insight:** KTO's strength is *not needing comparisons at all*. If you're already ranking things on a scale, you might as well use DPO.

### 4.4 GRPO: Group Relative Policy Optimization

GRPO (used in DeepSeek-V2/V3) is particularly interesting for reasoning tasks. It takes a fundamentally different approach from DPO:

**How it works:** For each prompt, sample a **group** of $G$ responses from the current policy. Score each response (using a rule-based verifier for math/code, or any reward function). Then compute **group-relative advantages** — each response is judged relative to others in its group, not against an absolute standard.

**Why this is clever for reasoning:** In math and coding, you often have a verifiable correct answer. Instead of training a noisy reward model, you can use exact-match verification (did the answer equal the ground truth?) or test-case pass rates. The group structure provides natural normalization: a response that scores 8/10 might be great (if the group averages 5/10) or terrible (if the group averages 9/10).

**Advantages over PPO for reasoning:** No critic (value function) needed. The group mean serves as the baseline. No reward model needed if you use rule-based verification. This is what enabled DeepSeek-R1 to achieve its breakthrough reasoning capabilities — the model learned to generate chain-of-thought reasoning traces entirely through GRPO, without any human annotations of reasoning steps.

```python
def grpo_advantages(rewards_per_group):
    """
    GRPO: compute relative advantages within each group.
    No critic, no reference model for the advantage computation.
    
    Args:
        rewards_per_group: (batch_size, group_size) tensor of rewards
    Returns:
        Normalized advantages within each group
    """
    # rewards shape: (batch_size, group_size)
    mean = rewards_per_group.mean(dim=1, keepdim=True)
    std = rewards_per_group.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards_per_group - mean) / std
    return advantages

# Example: 4 responses to a math problem, scored by answer correctness
rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0]])  # 2 correct, 2 wrong
advantages = grpo_advantages(rewards)
# Result: wrong answers get negative advantage, correct get positive
# The model learns to prefer what works within each group
```

![Figure 4: Alignment Methods Comparison](../zh/images/day15/alignment-methods-comparison.png)
*Caption: Comparison across key dimensions. Lower "Needs Ref Model" and "Needs Pairs" scores are better. Higher scores on stability, efficiency, and simplicity are better.*

---

## 5. When to Use What

### 5.1 Decision Framework

Here's a practical guide based on your situation:

| Situation | Recommended Method | Why |
|-----------|-------------------|-----|
| First-time alignment | DPO | Simplest, well-tested, good defaults |
| Only have binary labels | KTO | Doesn't need paired data |
| Limited compute / want simplicity | ORPO | One-stage, no reference model |
| Math / reasoning with verifiable answers | GRPO | Group comparison with rule-based rewards |
| Noisy preference data | IPO | Robust to label noise |
| Large-scale production (like ChatGPT) | PPO + RM | Still scales best with massive data |

### 5.2 The Meta-Trend

The field is clearly moving toward simpler methods. The progression tells the story:

![Figure 5: Alignment Timeline](../zh/images/day15/alignment-timeline.png)
*Caption: The rapid evolution of alignment methods from 2022-2024. Notice the trend: each new method requires fewer components.*

In 2022, you needed SFT + RM + PPO. By 2024, you can do alignment in a single stage (ORPO) or without paired data (KTO) or without any reference model (SimPO). Each year, the complexity decreases while results improve or match prior methods.

### 5.3 A Word on PPO's Continued Relevance

PPO isn't dead. For the largest production systems (GPT-4, Claude), PPO with reward models still offers advantages:

- **Online learning**: PPO can learn from new data during training; DPO is offline (fixed dataset)
- **Scalability**: With enough compute, PPO can iterate indefinitely
- **Nuanced reward shaping**: A reward model can capture complex trade-offs

But for most practitioners and researchers, the simplicity of DPO-family methods makes them the default choice in 2025.

---

## 6. Common Misconceptions

### ❌ "DPO is always better than PPO"

Not true. DPO is simpler and more stable, but PPO can achieve better results with enough engineering effort and data. The "better" method depends on your constraints.

### ❌ "You don't need a reference model with DPO"

You do need a frozen reference model — it's how DPO measures how much the policy has changed. This doubles memory compared to SFT alone.

### ❌ "DPO eliminates the need for good data"

DPO eliminates the *reward model*, not the *data quality requirement*. Garbage preference data will produce a garbage aligned model, regardless of the optimization method.

### ❌ "All these methods are fundamentally different"

Most DPO variants share the same core idea: measure the log-probability ratio between policy and reference, then push chosen responses up and rejected responses down. The differences are in the loss function shape and data requirements.

---

## 7. Further Reading

### Foundational Papers
1. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — The original DPO paper (Rafailov et al., 2023)
2. [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) — IPO framework (Azar et al., 2023)
3. [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — KTO with binary labels

### Newer Methods
4. [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Single-stage alignment (2024)
5. [SimPO: Simple Preference Optimization with Reference-Free Reward](https://arxiv.org/abs/2405.14734) — No reference model needed (2024)
6. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — Introduces GRPO

### Analyses and Comparisons
7. [Insights into Alignment: The Necessity of RLHF](https://arxiv.org/abs/2404.09643) — When RLHF is still worth the complexity

---

## Reflection Questions

1. If DPO implicitly recovers the reward model, why can't we just extract and use that reward model separately? What would break?
2. KTO uses prospect theory's insight that "losses loom larger than gains." How might this asymmetry affect the kinds of behaviors an aligned model learns?
3. GRPO uses group-relative rewards instead of absolute rewards. What kinds of tasks would benefit most from this relative comparison?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| DPO | Directly optimize policy from preferences — no reward model needed |
| IPO | Like DPO but with squared loss — more robust to noisy preferences |
| KTO | Only needs "good/bad" labels, not preference pairs — cheaper data |
| ORPO | Combines SFT + alignment in one stage — simplest pipeline |
| GRPO | Group-relative optimization — excels at reasoning with verifiable answers |
| Beta ($\beta$) | Controls how much the policy deviates from the reference model |
| Reference model | Frozen copy of SFT model — needed to measure policy drift |
| Reward hacking | When a policy exploits reward model gaps — impossible in DPO |

**Key Takeaway**: The alignment field has undergone a dramatic simplification since 2023. DPO proved you don't need reinforcement learning to align language models — just well-chosen preference data and a clever loss function. The newer alternatives (KTO, ORPO, GRPO) push this further: less data, fewer models, simpler code. For most practitioners, DPO-family methods should be the default; PPO remains relevant only for the largest-scale production systems with extensive engineering resources.

---

*Day 15 of 60 | LLM Fundamentals*
*Word count: ~2600 | Reading time: ~13 minutes*
