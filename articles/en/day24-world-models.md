# Day 24: World Models — Learned Simulators of Reality

> **Core Question**: What does it mean for a system to *model* the world rather than merely describe it, and has this become the central architectural question in AI?

---

## Opening

A toddler who has never seen a physics textbook still knows that a ball rolling behind a sofa continues to exist. She can predict, roughly, where it will emerge. She has, in some sense, a *model* of object permanence and physical dynamics — not a symbolic one, not one she can articulate, but one that supports prediction and action.

The question of whether machine learning systems can acquire something analogous — an internal predictive model of how states of the world evolve under actions and time — has become one of the defining questions of the field. Not because it is new (control theorists and roboticists have studied state-space models for decades), but because two major research programs now converge on it from opposite directions. On one side, the reinforcement learning community has spent the last decade building increasingly sophisticated learned simulators — *world models* — that enable planning, imagination, and sample-efficient learning. On the other side, large language models trained on text alone appear, to some researchers, to have implicitly learned rich world structure through the proxy of next-token prediction.

The debate is not merely philosophical. If LLMs already contain latent world models, then scaling language modeling further may be the most direct path to general intelligence. If they do not — if there is a fundamental gap between modeling the distribution of text and modeling the dynamics of the world — then we may need explicit architectural commitments to world modeling, and the question becomes: what form should those commitments take?

This article treats world models as a rigorous research topic, not a buzzword. We trace the intellectual lineage from classical control to modern latent dynamics models, work through the actual mathematics of how they are trained, examine the architectural innovations that make them work, and then confront the hard question: how do learned world models compare to what LLMs implicitly acquire?

---

## 1. Academic Lineage: From State Estimation to Learned Simulators

### 1.1 Classical roots

The idea that an agent should maintain an internal model of its environment predates machine learning entirely. In classical control theory, a *state-space model* consists of:

$$s_{t+1} = f(s_t, a_t) + \epsilon_t, \quad o_t = g(s_t) + \eta_t$$

where $s_t$ is the (possibly partially observed) state, $a_t$ is a control input, $o_t$ is an observation, and $\epsilon_t, \eta_t$ are noise terms. The Kalman filter (Kalman, 1960) provides the optimal recursive estimator for linear-Gaussian instances of this model. The entire framework presumes that someone — an engineer — has specified $f$ and $g$ by hand.

**Understanding the Kalman filter.** The Kalman filter operates in two alternating steps. In the *predict* step, the previous state estimate is propagated forward through the dynamics model to produce a prior. In the *update* step, a new observation is used to correct this prior, producing a posterior. The fusion is controlled by the **Kalman gain**:

$$\hat{s}_{t|t} = \hat{s}_{t|t-1} + K_t\,(o_t - \hat{o}_{t|t-1})$$

Here $\hat{s}_{t|t-1}$ is the predicted state (prior), $\hat{s}_{t|t}$ is the corrected state (posterior), $o_t$ is the observation, and $\hat{o}_{t|t-1}$ is the predicted observation. The term $(o_t - \hat{o}_{t|t-1})$ is the *innovation* or residual — the gap between what the sensor reports and what the model expected. The Kalman gain $K_t$ is computed automatically from the covariance matrices: when the model's uncertainty is large relative to the sensor's, $K_t$ is large (trust the sensor more); when the sensor is noisy, $K_t$ is small (trust the model more). This is not a hyperparameter to tune — it falls out of the math.

**Why the linear-Gaussian assumption matters.** A natural question is: if we already have observations, why bother with predictions at all? There are several reasons. Observations are noisy (a GPS reading might be off by 10 meters), partial (a sensor gives position but not velocity), and sometimes missing entirely (GPS drops in a tunnel). The prediction step provides a principled prior that the update step then refines.

But the linear-Gaussian assumption is what makes the whole thing *tractable*. The key property is **closure under linear transformation**: if $x \sim \mathcal{N}(\mu, \Sigma)$ and $y = Ax + b$, then $y \sim \mathcal{N}(A\mu + b, A\Sigma A^\top)$. A Gaussian stays Gaussian after linear transformation. This means that throughout the entire predict-update cycle — linear transition, add Gaussian noise, observe with a linear sensor, add more Gaussian noise — the posterior distribution over the state remains Gaussian at every timestep. You only need to track two quantities: the mean and the covariance. Once the system is nonlinear or the noise is non-Gaussian, the posterior may become multi-modal or skewed, and no simple recursive formula exists — you must resort to approximations like extended Kalman filters, particle filters, or, in the modern deep learning setting, variational inference with neural networks. This is precisely the trajectory from classical control to modern world models.

### 1.2 Model-based reinforcement learning

The transition to *learned* models began in earnest with Sutton's Dyna architecture (Sutton, 1991). Dyna interleaves real experience with *simulated* experience: the agent learns both a value function and a one-step model $\hat{P}(s_{t+1} | s_t, a_t)$, then uses the model to generate imaginary rollouts for additional policy updates. The insight was fundamental: a model, even an imperfect one, can massively amplify the value of each real interaction.

Subsequent work tightened the loop. PILCO (Deisenroth & Rasmussen, 2011) used Gaussian processes as forward models, propagating uncertainty through predicted trajectories to achieve remarkable sample efficiency on low-dimensional control tasks. PETS (Chua et al., 2018) replaced GPs with ensembles of probabilistic neural networks, scaling to higher dimensions while retaining uncertainty-aware planning via model-predictive control (MPC).

### 1.3 Ha & Schmidhuber (2018): The turning point

The paper simply titled *World Models* (Ha & Schmidhuber, 2018) reframed the conversation for the deep learning era. Their system had three components: a VAE encoder compressing visual observations into latent vectors; an MDN-RNN predicting the next latent state given the current latent state and action; and a small controller mapping latent states to actions. The key result was that the controller could be trained *entirely inside the model's own dreams* — hallucinated trajectories sampled from the learned dynamics — and then transferred to the real environment.

This was vivid but architecturally limited. The VAE and RNN were trained separately, the latent space was purely stochastic with no deterministic path, and there was no variational inference over full sequences. But the conceptual impact was enormous: it showed that a learned simulator could replace a hand-engineered one, and that policies trained in imagination could work in reality.

### 1.4 Dreamer: Recurrent State-Space Models

The Dreamer line of work (Hafner et al., 2020; 2023) represents the current state of the art in learned world models for control. The core innovation is the **Recurrent State-Space Model (RSSM)**, which combines a deterministic recurrent path (a GRU or similar) with a stochastic latent path:

$$\bar{s}_t = f_\theta^{\text{det}}(\bar{s}_{t-1}, s_{t-1}, a_{t-1})$$
$$s_t \sim q_\theta(s_t | \bar{s}_t, o_t)$$

Here $\bar{s}_t$ is the deterministic state (carrying long-term memory through recurrence), $s_t$ is the stochastic state (capturing uncertainty about the current situation), and $q_\theta$ is an approximate posterior inferred from the current observation. The transition prior is $p_\theta(s_t | \bar{s}_t)$.

Dreamer trains the world model end-to-end via variational inference, then learns a policy and value function entirely in latent space using imagined rollouts. DreamerV2 (2021) introduced discrete latents via straight-through gradients; DreamerV3 (2023) demonstrated that a single fixed hyperparameter configuration achieves state-of-the-art across 150+ diverse benchmarks — a level of generality previously unseen in model-based RL.

### 1.5 LeCun's JEPA and beyond

Yann LeCun's Joint-Embedding Predictive Architecture (JEPA) proposal (LeCun, 2022) reframes world modeling away from pixel-level reconstruction. A JEPA predicts the *representation* of the next state rather than the next observation itself:

$$z_{t+1} = P_\theta(z_t, a_t)$$

where $z_t = \text{Enc}(o_t)$ and the predictor $P_\theta$ operates in embedding space. A critic network enforces that predicted embeddings lie on the manifold of valid representations, replacing reconstruction with a consistency constraint. The motivation is clear: much of what we perceive is irrelevant to planning, and forcing a model to reconstruct pixels wastes capacity on details that do not matter for decision-making.

The trajectory is clear: **hand-crafted simulators → learned dynamics with fixed representations → end-to-end learned latent simulators → prediction in abstract embedding spaces**. Each step trades generality for efficiency, and the current frontier asks whether this progression can merge with the capabilities of large-scale foundation models.

---

## 2. Core Formulation: Variational Inference over Latent Dynamics

### 2.1 The generative model

A world model defines a generative process over sequences of observations $o_{1:T}$, actions $a_{1:T}$, and rewards $r_{1:T}$. The core latent variable model is:

$$p_\theta(o_{1:T}, r_{1:T}, s_{1:T} | a_{1:T}) = \prod_{t=1}^{T} p_\theta(o_t | s_t) \, p_\theta(r_t | s_t) \, p_\theta(s_t | s_{t-1}, a_{t-1})$$

where $s_t$ are latent states, $p_\theta(o_t | s_t)$ is the observation model (decoder), $p_\theta(r_t | s_t)$ is the reward model, and $p_\theta(s_t | s_{t-1}, a_{t-1})$ is the transition model. For the RSSM, the transition prior decomposes as $p_\theta(s_t | \bar{s}_t)$ where $\bar{s}_t$ is the deterministic recurrent state.

### 2.2 Training via the ELBO

We cannot compute the true posterior $p(s_{1:T} | o_{1:T}, a_{1:T})$, so we introduce an approximate posterior $q_\theta(s_{1:T} | o_{1:T}, a_{1:T})$ — typically factorized as $\prod_t q_\theta(s_t | \bar{s}_t, o_t)$ — and maximize the evidence lower bound (ELBO):

$$\mathcal{L} = \mathbb{E}_{q(s_{1:T})} \left[ \sum_{t=1}^{T} \left( \log p_\theta(o_t | s_t) + \log p_\theta(r_t | s_t) \right) \right] - \sum_{t=1}^{T} \text{KL}\left( q_\theta(s_t | \bar{s}_t, o_t) \, \| \, p_\theta(s_t | \bar{s}_t) \right)$$

This decomposes into three interpretable terms:

- **Reconstruction loss**: $\log p_\theta(o_t | s_t)$ — can the latent state reconstruct the observation?
- **Reward prediction**: $\log p_\theta(r_t | s_t)$ — does the latent state contain enough information to predict reward?
- **KL regularization**: $\text{KL}(q \| p)$ — keeps the approximate posterior close to the transition prior, preventing the model from memorizing observations into the posterior without learning good dynamics.

This is *not* simply "predict the next state." It is approximate Bayesian inference: the model maintains a *belief* over latent states, updated by observations via the encoder and regularized by the learned dynamics. The KL term is what makes this a principled probabilistic model rather than a deterministic autoencoder with a prediction head.

### 2.3 Deterministic vs. stochastic: why RSSM matters

Earlier models (Ha & Schmidhuber, 2018) used purely stochastic transitions. This created a problem: stochastic state alone forgets slowly. Gradients through sampling are noisy, and long-range dependencies degrade. The RSSM's innovation is the deterministic path $\bar{s}_t = f(\bar{s}_{t-1}, s_{t-1}, a_{t-1})$, which provides a stable memory channel. The stochastic state $s_t$ then only needs to capture *what is uncertain* about the current situation, conditioned on the deterministic summary. This separation is analogous to the way Kalman filters maintain both a state estimate and a covariance matrix — one for the best guess, one for the uncertainty.

### 2.4 The full Dreamer-style loss

In practice, DreamerV3 optimizes:

$$\mathcal{L}_{\text{world}} = \mathbb{E}_q \left[ \sum_t \beta_o \log p(o_t|s_t) + \beta_r \log p(r_t|s_t) + \beta_c \log p(\text{cont}_t|s_t) \right] - \beta_{\text{kl}} \sum_t \text{KL}(q_t \| p_t)$$

where $\text{cont}_t$ is a continuation flag (episode not done), and the $\beta$ coefficients are tuned to balance terms. Notably, DreamerV3 uses *discrete* latents (categorical distributions with straight-through gradients) and symlog predictions for the reward and continuation heads to handle non-Gaussian distributions gracefully.

---

## 3. Architecture Deep Dive

![Figure: World model architecture stack](../zh/images/day24/world-model-stack.png)
*A world model architecture: encoder maps observations to latent states; the RSSM maintains deterministic and stochastic state paths; reward and continuation heads predict task signals; a decoder optionally reconstructs observations.*

### 3.1 Encoder

The encoder maps a high-dimensional observation $o_t$ (image, point cloud, etc.) to parameters of the approximate posterior $q_\theta(s_t | \bar{s}_t, o_t)$. In DreamerV3, this is a shallow CNN followed by a linear layer producing logits for a categorical distribution over a $32 \times 32$ discrete latent. The encoder is essentially a VAE-style inference network, conditioned on the deterministic state $\bar{s}_t$ to provide temporal context.

Alternatives include continuous Gaussian latents (DreamerV1/V2), VQ-VAE discretization, or patch-based encoders borrowed from vision transformers.

### 3.2 Transition model (RSSM)

The heart of the system. At each step:

1. **Deterministic update**: $\bar{s}_t = \text{GRU}(\bar{s}_{t-1}, \text{concat}(s_{t-1}, a_{t-1}))$
2. **Prior**: $p_\theta(s_t | \bar{s}_t)$ — what the model *expects* before seeing the observation
3. **Posterior**: $q_\theta(s_t | \bar{s}_t, o_t)$ — what the model *infers* after seeing the observation
4. **KL divergence** between prior and posterior measures the *surprise* of the observation

During imagination (no observation available), the prior is used as the state, enabling rollouts without grounding.

### 3.3 Reward and continuation heads

Small MLPs map the latent state $(\bar{s}_t, s_t)$ to a predicted reward $\hat{r}_t$ and continuation probability. These are trained as part of the world model loss. The reward head is critical for planning: it tells the agent which imagined futures are desirable.

### 3.4 Decoder

The decoder reconstructs $o_t$ from $s_t$, providing the reconstruction loss term. Some recent work argues the decoder is unnecessary — if the model predicts rewards and dynamics well in latent space, reconstructing pixels is wasted capacity (cf. JEPA). In practice, the decoder acts as a regularizer: it forces the latent state to retain enough information to reconstruct the observation, which prevents the model from collapsing to a trivial representation. DreamerV3 retains the decoder but with a small $\beta_o$.

### 3.5 Planner / Policy

Dreamer learns a separate actor-critic network that operates *entirely in latent space*. The actor maps latent states to actions; the critic estimates the value function. Both are trained on imagined rollouts:

1. Imagine $H$ steps forward using the RSSM prior
2. Compute imagined returns using the reward head and critic
3. Update actor to maximize returns via policy gradients
4. Update critic via temporal-difference learning on imagined trajectories

Alternative planning approaches include Cross-Entropy Method (CEM) for shooting-based trajectory optimization (used in PETS, TD-MPC) or MPC with learned dynamics. The tradeoff: learned policies are fast at inference time but require training; CEM/MPC can adapt online but are computationally expensive per decision.

---

## 4. What World Models Buy You That LLMs Don't

![Figure: Next-token prediction vs world dynamics](../zh/images/day24/next-token-vs-world-dynamics.png)
*Comparing the training objectives: next-token cross-entropy vs state-transition ELBO.*

The "word model vs world model" framing is catchy but shallow. The real differences are structural and run far deeper than a pun.

### 4.1 Training objective

An LLM is trained to minimize cross-entropy loss on next-token prediction:

$$\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})$$

A world model is trained to maximize the ELBO over latent state sequences:

$$\mathcal{L}_{\text{WM}} = \mathbb{E}_q \left[ \sum_t \log p(o_t|s_t) + \log p(r_t|s_t) \right] - \text{KL}(q \| p)$$

These are not just different losses — they define different *modeling targets*. The LLM loss is purely discriminative: which token follows? The world model loss is generative: what is the latent state, and does it coherently explain observations, rewards, and dynamics? The ELBO enforces that the latent space has consistent geometry — states that are close in latent space produce similar futures — something no token-level objective guarantees.

### 4.2 Information-theoretic argument

Language is a *lossy, observational channel* for world state. When a text describes "the cat knocked the vase off the table," it compresses a high-dimensional physical event — trajectories, masses, friction, glass fragments — into a handful of tokens. Training on text means your signal about world dynamics passes through this bottleneck. A world model, by contrast, can be trained on raw sensorimotor data (pixels, joint angles, point clouds) that preserves the full information content of the interaction. The modeling target is the generative process itself, not a linguistic summary of its outputs.

### 4.3 Action conditioning

World models are explicitly conditioned on actions: the transition function takes $(s_t, a_t)$ as input. This is not an afterthought — it is the core of the architecture. The model *must* predict what happens *if* the agent does $a_t$, which is precisely what planning requires. LLMs receive action information only incidentally through text (e.g., "I moved the knight to e4"). There is no architectural mechanism that forces an LLM to maintain counterfactual state estimates under different action sequences.

### 4.4 Latent state consistency

World models enforce state persistence through the RSSM: $\bar{s}_t$ is a deterministic function of the entire history $(\bar{s}_{0:t-1}, s_{0:t-1}, a_{0:t-1})$. This means the latent state is a *sufficient statistic* for the history (approximately, modulo stochastic state). LLMs have no explicit state variable — the entire history must be reconstructed from the context window. This works remarkably well within the context window but degrades catastrophically when the window is exceeded, and there is no principled mechanism for maintaining a persistent belief state across turns without the KV cache acting as a surrogate.

### 4.5 Compounding error

All autoregressive models suffer from compounding error: errors at step $t$ become inputs to step $t+1$, and the distribution of rollouts diverges from reality. World model architectures address this in several ways:

- **KL regularization** forces the posterior to stay close to the prior, preventing the model from making brittle predictions that exploit specific observation details.
- **Deterministic paths** in the RSSM provide stable gradients across long horizons.
- **Short imagination horizons** with value function backups (Dreamer) avoid the need for very long accurate rollouts.

LLMs have no analogous mechanism. Chain-of-thought reasoning accumulates errors, and there is no latent variable that gets regularized toward a consistent dynamics model. The model simply conditions on its own outputs, errors and all.

### 4.6 Uncertainty representation

The stochastic component of the RSSM provides a natural mechanism for representing *epistemic uncertainty* — uncertainty about the model's own knowledge. When the model is in a novel situation, the posterior will be broad (high entropy), and the KL divergence will be large. This signal can be used for exploration, risk-sensitive planning, or detecting distribution shift. LLM confidence scores, by contrast, are poorly calibrated for state estimation. Token probabilities reflect the distribution of text, not the model's uncertainty about the underlying state of the world.

---

## 5. Where LLMs Genuinely Compete

Despite these structural advantages, dismissing LLMs as "merely word models" would be a mistake. Several lines of evidence suggest that LLMs acquire something like world structure through scale.

### 5.1 Emergent world modeling from scale

Probing studies have shown that LLM internal representations encode surprisingly rich structure. OthelloGPT (Li et al., 2023) demonstrated that a GPT-2 model trained only on Othello game transcripts develops internal representations that linearly encode the board state — despite never seeing the board. The model learned to maintain a world state because predicting the next move *requires* it, and the training distribution is clean enough that the signal is recoverable.

Similar findings have emerged for spatial reasoning (reading comprehension tasks that implicitly require spatial maps), temporal reasoning (tracking event sequences), and causal reasoning (identifying causal direction from correlational text).

### 5.2 Text as a surprisingly rich training signal

Text is not just a lossy compression of the world — it is a *highly curated* compression produced by intelligent agents who highlight causal, temporal, and spatial structure. A physics textbook, a news report, and a novel all encode world dynamics, just at different levels of abstraction. At sufficient scale, the statistical structure of text may be rich enough that a model trained to predict it *must* learn world structure as an intermediate representation — because doing so is the most efficient way to compress the training distribution.

### 5.3 The "latent world model" argument

The argument, stated formally: if $P(x_t | x_{<t})$ depends on some latent world state $s_t$, and the text distribution is rich enough that $s_t$ is (approximately) identifiable from $x_{<t}$, then the LLM's internal representation at layer $l$, position $t$, must encode something functionally similar to $s_t$ — because that is the most parameter-efficient way to implement the conditional distribution.

This is a plausible but unproven argument. The key question is *identifiability*: is the world state sufficiently determined by the text? In many domains (games, formal reasoning, simple physical scenarios), yes. In open-ended, partially observed, embodied settings — almost certainly not, or at least not without massive redundancy.

### 5.4 Limitations

The strongest counterarguments:

- **Coverage**: Text does not cover all states the agent may encounter, especially novel physical configurations.
- **Granularity**: Text describes the world at the level of objects, events, and relations, not at the level of continuous dynamics needed for control.
- **Action grounding**: Text describes actions after the fact; it does not provide the counterfactual experience of trying actions and observing outcomes.
- **Distribution shift**: LLMs are trained on human-written text. When an agent acts in the world and generates novel situations, the text distribution no longer covers the relevant state space.

---

## 6. Current Frontier: Convergence and Open Problems

![Figure: Planning in latent space](../zh/images/day24/planning-in-latent-space.png)
*Agents plan by imagining trajectories through learned latent dynamics before committing to actions in the real world.*

### 6.1 Video generation as world modeling

Sora (OpenAI, 2024), Genie (Bruce et al., 2024), and similar systems generate video conditioned on initial frames and (optionally) actions. To generate coherent video over many frames, the model must learn something about physical dynamics — object persistence, occlusion, gravity, collision. These systems can be seen as world models trained on passive (non-interactive) data, with the decoder being a video generator rather than a single-frame reconstructor.

The gap: these models are not yet action-conditioned at the granularity needed for control, and they lack the reward/value structure needed for planning. But the convergence direction is clear.

### 6.2 World models for LLM agents

Systems like Voyager (Wang et al., 2023) and DEPS (Wang et al., 2023) use LLMs as planners in Minecraft-like environments, maintaining explicit state libraries and skill libraries. The LLM generates plans in natural language, executes them, and receives text-based feedback from the environment. This is world modeling at the *linguistic level* — effective for high-level planning but brittle for fine-grained control.

More recent work explores training world models on *internet-scale data* (video, text, action logs) to create general-purpose simulators that LLM agents can query during planning. The vision: an LLM that can "imagine" physical outcomes before acting, using a learned world model as a mental simulator.

### 6.3 Multimodal foundation models with dynamics heads

An emerging architecture combines a large pretrained vision-language model with a dynamics prediction head. The VLM provides semantic understanding; the dynamics head learns to predict future states conditioned on actions. This hybrid approach sidesteps the question of whether text alone is sufficient by simply giving the model access to richer modalities.

### 6.4 Open problems

- **Scaling latent world models**: Current world models (DreamerV3) are trained from scratch on single environments. Scaling to internet-scale diverse data without losing the precision needed for control is an open challenge.
- **Grounding**: How do you ground a world model trained on passive video data so that it can be used for planning? Action labels are scarce.
- **Long-horizon reliability**: Compounding error in latent space remains the fundamental bottleneck. Can hierarchical or compositional world models maintain coherence over thousands of steps?
- **The architecture question**: Can a single architecture — presumably a large transformer — serve both as a language model and a world model, or are there fundamental tradeoffs between the distribution modeling required for language and the dynamics modeling required for planning?

![Figure: World model limitations](../zh/images/day24/world-model-limitations-radar.png)
*Radar chart of current world model limitations across key dimensions.*

---

## 7. Code Sketch: Dreamer-Style Planning in Latent Space

```python
import torch
import torch.nn as nn

class RSSM(nn.Module):
    """Simplified Recurrent State-Space Model."""
    def __init__(self, det_dim=200, stoch_dim=30, stoch_classes=32, action_dim=7):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        # Deterministic path: GRU
        self.gru = nn.GRUCell(det_dim, det_dim)
        # Prior: predict stochastic state from deterministic state
        self.prior_net = nn.Linear(det_dim, stoch_dim * stoch_classes)
        # Posterior: condition on deterministic state + encoded observation
        self.post_net = nn.Linear(det_dim + 256, stoch_dim * stoch_classes)
        # Action embedding
        self.action_embed = nn.Linear(action_dim, det_dim)

    def get_dist(self, logits):
        """Reshape logits to categorical distribution parameters."""
        return logits.view(-1, self.stoch_dim, self.stoch_classes)

    def forward(self, prev_det, prev_stoch, action, obs_embed=None):
        # Deterministic update
        x = prev_stoch.flatten(-2).mean(-1) if prev_stoch.dim() == 3 else prev_stoch
        gru_input = self.action_embed(action) + x
        det_state = self.gru(gru_input.unsqueeze(0), prev_det.unsqueeze(0)).squeeze(0)
        # Prior
        prior_logits = self.get_dist(self.prior_net(det_state))
        if obs_embed is None:
            # Imagination: use prior
            stoch_state = torch.softmax(prior_logits, -1)
        else:
            # Inference: use posterior
            post_logits = self.get_dist(self.post_net(
                torch.cat([det_state, obs_embed], -1)))
            stoch_state = torch.softmax(post_logits, -1)
        return det_state, stoch_state, prior_logits


class WorldModel(nn.Module):
    """Minimal world model with RSSM, reward head, and decoder."""
    def __init__(self):
        super().__init__()
        self.rssm = RSSM()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 512), nn.ELU(),
            nn.Linear(512, 256))
        self.reward_head = nn.Sequential(
            nn.Linear(200 + 30 * 32, 256), nn.ELU(),
            nn.Linear(256, 1))
        self.decoder = nn.Sequential(
            nn.Linear(200 + 30 * 32, 512), nn.ELU(),
            nn.Linear(512, 64 * 64 * 3))

    def imagine(self, initial_det, initial_stoch, policy, horizon=15):
        """Imagine a trajectory using the learned dynamics."""
        det, stoch = initial_det, initial_stoch
        rewards = []
        for _ in range(horizon):
            latent = torch.cat([det, stoch.flatten(-2).mean(-1)], -1)
            action = policy(latent)  # learned actor
            det, stoch, _ = self.rssm(det, stoch, action, obs_embed=None)
            reward = self.reward_head(torch.cat([det, stoch.flatten(-2).mean(-1)], -1))
            rewards.append(reward)
        return torch.stack(rewards)
```

This sketch shows the core loop: the RSSM maintains a deterministic state and samples a stochastic state; during imagination, the prior (no observation) is used; the reward head evaluates imagined futures; a policy learns to maximize imagined returns.

---

## 8. Common Misconceptions

**"World models are just simulators."** A simulator (e.g., MuJoCo, Unity) is hand-engineered with known physics. A learned world model discovers dynamics from data, including dynamics that are difficult or impossible to specify by hand (e.g., deformable objects, multi-agent interactions, visual appearance under novel lighting).

**"World models need pixel reconstruction."** The JEPA line of work explicitly argues against this. Prediction in latent space, with a consistency constraint, can be more efficient than reconstruction. The debate over whether reconstruction is necessary or merely convenient is ongoing.

**"LLMs already have world models, so this is moot."** Having representations that *probe* as world-state-like is not the same as having a dynamics model that supports reliable multi-step planning with action conditioning and uncertainty estimation. The gap may close with scale, but it has not closed yet.

**"World models solve the sample efficiency problem."** They help dramatically in domains where interaction is expensive, but they introduce their own failure mode: if the model is wrong, the policy trained in imagination will exploit the model's errors (the classic "model exploitation" problem in model-based RL).

---

## 9. Further Reading

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Sutton, "Dyna, an Integrated Architecture for Learning, Planning, and Reacting" | 1991 | First integration of learned model + planning in RL |
| Deisenroth & Rasmussen, "PILCO" | 2011 | GP-based model for sample-efficient policy search |
| Ha & Schmidhuber, "World Models" | 2018 | VAE + MDN-RNN; training in imagination |
| Hafner et al., "Dream to Control" (Dreamer) | 2020 | RSSM; end-to-end latent actor-critic |
| Hafner et al., "Mastering Diverse Domains" (DreamerV3) | 2023 | Fixed hyperparameters across 150+ benchmarks |
| LeCun, "A Path Towards Autonomous Machine Intelligence" (JEPA) | 2022 | Predict in embedding space, not observation space |
| Li et al., "OthelloGPT" | 2023 | Emergent board state in LLM trained on game transcripts |
| Bruce et al., "Genie" | 2024 | Interactive environment generation from video |
| OpenAI, "Sora" | 2024 | Video generation as implicit world modeling |

---

## Reflection Questions

1. If an LLM's internal representations linearly decode to world state (as in OthelloGPT), does that constitute a world model? What additional capabilities would it need before you would call it one?

2. The JEPA proposal argues against reconstruction. But without reconstruction, how do you prevent the latent space from collapsing? What constraints make prediction-in-latent-space non-trivial?

3. Consider the compounding error problem. If a world model's rollouts diverge from reality after $k$ steps, what are the implications for planning horizon $H$? How does the Dreamer architecture sidestep this?

4. Can a single transformer architecture jointly serve as a language model and a world model? What would the training objective look like? What are the fundamental tradeoffs?

---

## Summary

| Aspect | World Model (Dreamer-style) | LLM (next-token) |
|--------|----------------------------|-------------------|
| Training objective | ELBO: reconstruction + reward + KL | Cross-entropy on tokens |
| State representation | Explicit latent $s_t$ (deterministic + stochastic) | Implicit (context window + KV cache) |
| Action conditioning | Architecturally built-in | Incidental through text |
| Uncertainty | Modeled via stochastic latent | Token probabilities (poorly calibrated for state) |
| Planning | Imagined rollouts in latent space | Chain-of-thought / in-context reasoning |
| Compounding error | Mitigated by KL, deterministic path, short rollouts | No architectural mitigation |
| Data modality | Sensorimotor (pixels, actions, rewards) | Text (optionally multimodal) |
| Sample efficiency | High (reuses model for imagined experience) | High (due to internet-scale pretraining) |
| Generality | Limited per-environment training | Broad but shallow physical reasoning |

World models are not a competing paradigm to LLMs — they are a complementary one. The central question is not "which is better" but "how do we combine the broad knowledge of foundation models with the structured dynamics modeling of learned simulators?" The answer to that question will shape the next generation of AI systems.

---

*Day 24 of 60 | LLM Fundamentals*
