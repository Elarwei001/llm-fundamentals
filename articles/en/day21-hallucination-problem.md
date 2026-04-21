# Day 21: The Hallucination Problem

> **Core Question**: Why do large language models sometimes produce answers that sound confident, detailed, and professional, yet are simply false?

---

## Opening

If you ask a calculator what 17 × 19 is, you expect one of two outcomes. Either it returns the right answer, or it fails because the battery is dead. What you do **not** expect is a polished paragraph saying, with total confidence, that the answer is 347 because “historically this multiplication often rounds upward in practical settings.”

That weird third failure mode is what makes LLM hallucination so unsettling.

A hallucination is not just any mistake. Humans make mistakes too. A hallucination is a particular kind of model failure where the output is **fluent, plausible, and unsupported**. The model fills in missing information with something that statistically fits the context, even when the statement is wrong.

This matters because language models are optimized to continue text, not to maintain an internal truth ledger. If the most likely continuation in the training distribution sounds like a specific citation, a concrete date, or a confident explanation, the model may produce it even when it has no reliable basis.

Think of it like an improv actor on stage. The actor has learned thousands of patterns for keeping a scene going. When a gap appears, the actor’s instinct is not to stop and say, “I am uncertain.” The instinct is to keep the scene coherent. LLMs often behave the same way. Coherence and truth overlap a lot, but they are not the same objective.

In this article we will separate hallucination from ordinary error, explain the major causes, look at why confidence is a misleading signal, and study what actually helps in practice. The short version is important: hallucination is not one bug with one fix. It is the natural result of several design choices colliding, including next-token prediction, imperfect memory, noisy context, and imperfect decoding.

---

## 1. What counts as hallucination

**One-sentence summary**: Hallucination is best understood as a failure of grounding, where the model generates claims not adequately supported by the input, reliable memory, or external evidence.

People use the word loosely, so it helps to distinguish a few cases.

### 1.1 Ordinary error vs hallucination

An ordinary error can happen when the model miscalculates, misparses a question, or chooses a weak reasoning path. A hallucination is more specific. The answer contains invented or unjustified content that is presented as if it were known.

Examples:

- **Ordinary error**: The model solves a probability problem incorrectly because it mixes up conditional and marginal probability.
- **Hallucination**: The model cites a nonexistent paper, invents a benchmark result, or claims a person said something they never said.

This is why hallucination is especially dangerous in domains like law, medicine, compliance, and research. The output can look more trustworthy precisely because it contains extra details.

### 1.2 Factual, citation, and reasoning hallucinations

A practical taxonomy is:

1. **Factual hallucination**: false statements about the world, such as wrong biographies, dates, definitions, or numbers.
2. **Citation hallucination**: fabricated references, URLs, section numbers, quotes, or sources.
3. **Reasoning hallucination**: steps that look logical but are unsupported or internally inconsistent.
4. **Context hallucination**: the answer contradicts the provided document, user message, or tool result.

The important pattern is the same in each case. The model is producing a continuation that is linguistically smooth but epistemically weak.

![Figure 1: Confidence and truth are not the same thing](../zh/images/day21/confidence-vs-factuality.png)
*Caption: The painful failure mode is not low confidence and low accuracy. It is high confidence with low factual support.*

A useful phrase here is **factuality**. Factuality asks whether the answer matches the world or at least matches the trusted evidence available to the system. Fluency is cheap. Factuality is expensive.

---

## 2. Why next-token prediction creates pressure toward hallucination

**One-sentence summary**: The base training objective rewards producing likely continuations, not explicitly representing uncertainty or verifying truth against the outside world.

A language model is trained to minimize next-token prediction loss:

$$
\mathcal{L} = - \sum_{t=1}^{T} \log P(x_t \mid x_{1:t-1}).
$$

That objective is powerful because so much structure is hidden inside language. It teaches syntax, semantics, style, broad world knowledge, and many latent task patterns. But notice what it does **not** directly say. It does not say:

- only answer when you are certain,
- distinguish observed facts from guessed completions,
- consult an external database before naming an obscure number,
- or abstain gracefully when evidence is missing.

If the training data often contains a pattern like “Question -> specific answer,” then a specific answer may receive higher probability than a careful refusal, even when the model is operating near the boundary of what it actually knows.

> **Intuition:** the model may not be confidently correct, but it has learned a strong habit: when a question appears, a concrete answer usually follows. In the gray zone where it only half-knows the fact, producing a specific-looking answer can still look more like the training distribution than saying “I’m not sure.” In other words, the model is rewarded for sounding like an answer machine before it is rewarded for being epistemically cautious.

This is one reason a model may guess instead of saying “I don’t know.” During standard supervised language modeling, guessing can still be locally rewarded if the guess often resembles what answers look like in the corpus.

A good analogy is autocomplete on steroids. Autocomplete is excellent at producing the next likely phrase. But “likely continuation” and “truthful response to a novel question” are only partly aligned.

> **Intuition:** Sometimes the most likely continuation is also the true answer, as in “The capital of France is ... Paris.” But for a rare or novel question, the model may not really know the fact. It still has strong pressure to continue with something that *looks like an answer* — a date, a number, a name, a fluent sentence. So “what sounds like the next plausible phrase” and “what is actually true” overlap often, but they are not the same objective.

Some recent work frames the issue in terms of training and evaluation incentives. If systems are rewarded for producing complete-looking answers more than calibrated uncertainty, then confident guessing becomes rational under the objective, even if it is harmful for users.

---

## 3. Knowledge boundaries, stale memory, and missing world access

**One-sentence summary**: Even a very large model contains only compressed, partial, and time-bounded world knowledge, so many prompts push it outside its reliable knowledge boundary.

LLMs do not store facts the way a database does. They compress statistical patterns from huge corpora into parameters. This gives them impressive recall, but it has hard limits.

| Problem | What it means | Why it matters |
|---|---|---|
| **Compression is not exact storage** | Model weights contain a blurry, distributed memory, not an indexed table with provenance | Common stable facts are often reliable, but niche API changes, rare paper details, or exact citations are much weaker |
| **Knowledge can be outdated** | A pretrained model only knows what was present before its cutoff, plus what later tuning added | If the world changes after the cutoff, the model may confidently answer from stale information |
| **Long-tail facts are harder** | Rare facts, rare names, and edge-case details receive much weaker training support | Models are often good at broad explanations but weaker on obscure entities, exact references, and rapidly changing data |

You can think of the model’s knowledge boundary like the edge of a map. Near the center, the terrain is detailed. Near the edge, the model still tries to draw roads, but some of them are invented.

---

## 4. Context-level causes, including retrieval and prompt issues

**One-sentence summary**: Hallucination is often a system problem, not just a base-model problem, because bad prompts, poor retrieval, and noisy context all degrade grounding.

Many production hallucinations happen even when the base model is decent. The larger system feeds it weak context.

Common causes include:

1. **Ambiguous prompts**: the model does not know whether to summarize, infer, speculate, or answer only from supplied evidence.
2. **Missing constraints**: if you do not say “use only the document,” the model may mix retrieved evidence with prior memory.
3. **Poor retrieval**: Retrieval-Augmented Generation (RAG) helps only if the retrieved chunks are relevant, readable, and sufficient.
4. **Context overload**: in long prompts, key evidence may be buried or contradicted by distractors.
5. **Tool-output mismatch**: the system may retrieve evidence in one format while the model is prompted to answer in another, causing brittle synthesis.

| Explanation | Figure |
|---|---|
| **What do “decoding effects” mean?**<br><br>Even if the model’s internal probability distribution is already a bit shaky, the **decoding strategy** is what decides which token actually gets emitted. This includes choices like greedy decoding, temperature, top-k, and top-p sampling.<br><br>If the distribution contains some low-quality but plausible-looking continuations, an aggressive sampling strategy can amplify them. In other words, the model may already be uncertain, and decoding decides whether that uncertainty turns into a cautious answer or a confident-looking hallucination.<br><br>So decoding effects are not always the root cause. They often act as an **amplifier**: upstream problems create weak candidates, and decoding determines whether one of those weak candidates becomes the actual output and then snowballs. | ![Figure 2: Several paths can lead to the same symptom](../zh/images/day21/hallucination-causes-map.png)<br><br>*Hallucinated answers are usually the visible symptom of multiple upstream problems, not one isolated defect.* | 

This is why saying “just use RAG” is too shallow. Retrieval reduces hallucination only when three conditions hold:

- the right evidence is retrieved,
- the model actually uses it,
- and the prompt makes unsupported claims less attractive than grounded ones.

If retrieval fetches a semantically similar but wrong document, the model may become **more** confidently wrong because it now has supporting-looking text.

---

## 5. Exposure bias and self-amplifying mistakes

**One-sentence summary**: During training the model sees true previous tokens, but during inference it conditions on its own generated tokens, so one early mistake can snowball.

Another structural issue is the gap between training and inference.

During training, models often use **teacher forcing**: when predicting token $x_t$, the model is conditioned on the true prefix $x_{1:t-1}$. During inference, it must condition on its own previous outputs.

That means the distribution at generation time is different from the distribution at training time. If the model makes one unsupported claim early, that claim becomes part of the context for later tokens. The model may then elaborate on it, adding fake specifics, invented citations, and fabricated causal explanations.

![Figure 3: Why one wrong token can poison the rest of the answer](../zh/images/day21/teacher-forcing-vs-inference.png)
*Caption: Inference is fragile because the model must continue from its own guesses rather than a guaranteed-correct prefix.*

This is sometimes discussed under **exposure bias**. The idea is simple. The model is exposed during training to clean prefixes more often than to its own imperfect generations. So at test time, it may respond poorly to the kind of corrupted context it created itself.

> **A useful intuition:** in effect, the model gets pulled into a pattern of *making the current story hang together*. Once a wrong claim enters the prefix, the next-token objective rewards local coherence, so later tokens often continue to support that claim rather than stop and correct it. The model is not deliberately lying, but it can behave as if it is trying to "make its story self-consistent."

This helps explain why hallucinations often come in clusters. Once the answer starts drifting, later details may be built on top of an already false premise.

---

## 6. Why confidence is a bad safety signal

**One-sentence summary**: The style of an answer is not a reliable estimate of its truth, because models are optimized for plausible continuation rather than calibrated epistemic reporting.

Humans are easily fooled by tone. Specific numbers, technical vocabulary, citations, and neat structure create an illusion of reliability. But none of those features guarantee support.

This is why “the answer sounded convincing” should never be part of your evaluation rubric.

In probabilistic terms, the model is good at assigning probability to text continuations, not necessarily at producing a calibrated estimate of whether a proposition is true in the world. Those are related but different tasks.

A subtle but important point is that instruction tuning can improve helpfulness and honesty on average, but it can also make answers *more polished*. So a stronger conversational model can sometimes feel safer while still hallucinating in edge cases.

For users, the operational lesson is blunt: **confidence must be earned by grounding, not inferred from prose quality**.

---

## 7. What actually reduces hallucination

**One-sentence summary**: The strongest mitigations combine better prompting, external grounding, uncertainty handling, and post-generation verification.

There is no universal fix, but there is a reliable hierarchy of interventions.

| Mitigation | What it does | Why it helps |
|---|---|---|
| **Better prompts and better task framing** | Reduce ambiguity, require evidence use, separate facts from assumptions, request uncertainty fields | Often reduces unnecessary speculation, though it does not solve hallucination by itself |
| **Retrieval and tool use** | Bring in search, databases, documentation lookup, calculators, and other tools | Gives the model contact with the outside world rather than relying only on parametric memory |
| **Verification and cross-checking** | Run a second pass to check claims against sources, schemas, or explicit evidence | Catches unsupported claims after generation |
| **Abstention and calibrated refusal** | Allow the system to say “I don’t know” or provide a partial answer with uncertainty markers | Safer than forcing a confident full answer in high-risk domains |
| **Training-time improvements** | Fine-tune on grounded data, encourage honesty, or optimize for factuality-aware behavior | Helps on average, but still complements rather than replaces good system design |

![Figure 4: The mitigation ladder](../zh/images/day21/mitigation-ladder.png)
*Caption: Prompting helps, but grounding and verification usually have larger effects on factual reliability.*

A practical slogan is: **prompting shapes behavior, retrieval supplies evidence, verification enforces discipline**.

---

## 8. Common misconceptions

**One-sentence summary**: Hallucination is often misunderstood as either a solved problem or proof that LLMs are useless, and both extremes are wrong.

### Misconception 1: “Hallucination means the model is broken.”

Not exactly. Hallucination is a predictable consequence of using a probabilistic language model for tasks that users interpret as truth-seeking. The model can still be extremely useful if the task and safeguards fit the failure mode.

### Misconception 2: “A better prompt can eliminate hallucination.”

No. Prompting can reduce some failure modes, especially context confusion, but it cannot give the model real-time world knowledge or perfect verification.

### Misconception 3: “RAG solves hallucination.”

Also no. RAG reduces one class of failure by providing evidence, but retrieval itself can fail and the model can still misread, ignore, or overgeneralize from the evidence.

### Misconception 4: “Hallucination only matters for factual Q&A.”

It matters anywhere unsupported details are costly. That includes summaries, code generation, compliance reports, clinical notes, and agent tool use.

---

## 9. Practical design rules for builders

**One-sentence summary**: If you are building with LLMs, assume hallucination will happen and design the product so that unsupported content is visible, limited, and recoverable.

Here are concrete rules that work better than vague hope:

1. **Match the tool to the task**. Use calculators for arithmetic, databases for exact records, retrieval for dynamic facts, and LLMs for synthesis and language.
2. **Require evidence where possible**. Ask for citations, quoted spans, or source IDs tied to each key claim.
3. **Separate generation from verification**. One pass writes, another pass checks.
4. **Prefer abstention over invention** in high-risk settings.
5. **Evaluate on realistic failure cases**. Include stale facts, adversarial ambiguity, and missing-evidence prompts.
6. **Track unsupported claims, not just task completion**. A fluent lie that completes the workflow is still a bad outcome.

If you remember only one engineering lesson, let it be this: treat the model like a brilliant intern with no built-in right to invent missing facts. Give it documents, tools, constraints, and a reviewer.

---

## Optional Math: A simple view of why retrieval helps

Suppose the model answers from a mixture of two information sources: parametric memory $M$ and retrieved evidence $R$. A cartoon version of the answer distribution is

$$
P(y \mid q, R) \approx \lambda \; P(y \mid q, R, M) + (1-\lambda) \; P(y \mid q, M),
$$

where $q$ is the question and $\lambda$ reflects how strongly the system uses retrieved evidence.

If retrieval is relevant and the prompt emphasizes grounding, increasing $\lambda$ should reduce unsupported claims. But if retrieval is noisy or irrelevant, a larger $\lambda$ may not help. In other words, retrieval is not magic. It only helps when the evidence pipeline is good.

---

## Further Reading

- Lewis et al., **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (2020), for the classic RAG formulation.
- Huang et al., **A Survey on Hallucination in Large Language Models** (2024/2025 versions), for a broad taxonomy and mitigation overview.
- OpenAI, **Why Language Models Hallucinate** (2025), for an argument about training incentives and uncertainty.

---

## Reflection Question

If an LLM product gives a very polished answer with a citation, which would you trust more: the writing style or the evidence path? Why?

---

## Closing

Hallucination is not an embarrassing side bug that will disappear just because models get larger. It sits near the center of what language models are: systems trained to continue text fluently under uncertainty.

That sounds pessimistic, but it is actually clarifying. Once you stop expecting raw language generation to behave like a verified database, good design choices become obvious. Ground the model. Constrain the task. Verify important claims. Allow uncertainty. Build products that make evidence visible.

Tomorrow we will stay on the same theme of capability boundaries, but from a different angle: reasoning. When a model writes a multi-step argument, is it actually reasoning, or just producing text that looks like reasoning? That question turns out to be trickier than it first appears.
