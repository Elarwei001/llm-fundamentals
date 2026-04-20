# Day 20: Prompt Engineering

> **Core Question**: If LLMs are already pretrained on the whole internet, why can a few lines of prompt wording change the answer so much?

---

## Opening

Prompt engineering sounds trivial at first. You type a request, the model answers, end of story. That view misses what is really happening. A prompt is not just a question. It is the entire temporary workspace that conditions the model's next-token distribution.

In yesterday's article we talked about the **context window**. Prompt engineering is the art of using that context window well. You are deciding what instructions, examples, constraints, and evidence enter the model's short-term working memory, and in what order.

A useful mental model is this: **pretraining gives the model a huge compressed prior, while the prompt chooses which part of that prior becomes active right now**. The weights are the library. The prompt is the librarian saying, "bring me the books about fraud detection, answer in JSON, be concise, and follow these examples."

That is why prompts matter so much. They do not usually add new knowledge to the model, but they strongly shape retrieval, attention allocation, reasoning style, and output format.

Think of prompt engineering like briefing a new but very capable teammate. If you say, "handle this," you may get something vaguely useful. If you say, "summarize these logs, extract only user-visible incidents, rank severity from 1 to 5, and return a markdown bullet list," the odds of a reliable answer go up sharply.

This article explains what prompt engineering really is, why it works, the main design patterns, where it breaks, and what changed as models became better instruction followers.

---

## 1. What prompt engineering actually changes

**One-sentence summary**: Prompt engineering modifies the conditional distribution over outputs by changing the context the model attends to.

An autoregressive language model produces tokens according to

$$
P(y \mid x) = \prod_{t=1}^{T} P(y_t \mid x, y_{1:t-1}),
$$

where $x$ is the prompt and $y$ is the generated answer. Prompt engineering changes $x$, so it changes every later conditional probability.

That sounds obvious, but it has an important consequence. The model is not selecting from one fixed answer hidden inside its weights. It is producing a distribution over many plausible continuations. Small prompt changes can move probability mass toward a different style, different reasoning path, or different final answer.

This is why the same model may act like:

- a concise API returning strict JSON,
- a patient tutor explaining from first principles,
- a careless rambler when instructions are vague,
- or a highly structured agent that calls tools in the right order.

All of those behaviors are latent possibilities. The prompt determines which one becomes likely.

![Figure 1: Prompt anatomy](../zh/images/day20/prompt-anatomy.png)
*Caption: Strong prompts usually contain clear structure: role, task, context, constraints, and output shape.*

A practical way to think about prompts is to separate five common components:

1. **Role**: Who should the model act as, if any?
2. **Task**: What exactly should it do?
3. **Context**: What evidence, data, or background should it use?
4. **Constraints**: What rules must it obey?
5. **Output format**: What shape should the answer have?

Many prompt failures come from missing one of these blocks. If the task is clear but the output format is not, the answer may be correct but unusable. If the format is clear but the evidence is missing, the answer may be cleanly wrong.

---

## 2. Why prompts work at all

**One-sentence summary**: Prompts work because Transformers are trained to continue text, and the prompt defines the textual situation the model believes it is in.

During pretraining, the model sees countless patterns such as instructions followed by answers, documents followed by summaries, bug reports followed by diagnoses, and question-answer pairs in many formats. It learns that different textual setups imply different likely continuations.

So when you write a prompt, you are not sending commands to a traditional symbolic program. You are creating a *pattern-completion environment*.

If the prompt looks like a tutoring session, the model is more likely to continue in tutoring style. If it looks like examples of labeled data, the model is more likely to infer the labeling rule. If it looks like a JSON API contract, the model is more likely to emit JSON.

That is why prompt engineering often feels a little like programming and a little like psychology. You are not changing weights directly, but you are shaping the attention landscape inside the forward pass.

Mathematically, attention weights depend on token interactions:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

If you change the prompt, you change the keys and queries available to later tokens. In plain English, you change which prior instructions, examples, and facts are easy for the model to focus on.

This is why wording, ordering, delimiters, and examples all matter. They are not magical incantations. They are ways of making the right patterns salient.

---

## 3. In-context learning, the engine under prompt engineering

**One-sentence summary**: Prompt engineering relies heavily on in-context learning, where the model infers a temporary task from examples in the prompt.

One of the most surprising discoveries in large models is that they can often learn a task from demonstrations inside the prompt, without updating weights. This is called **in-context learning**.

For example, suppose your prompt contains:

- `great -> positive`
- `terrible -> negative`
- `boring -> negative`
- `excellent -> ?`

The model may infer that the task is sentiment classification and output `positive`.

![Figure 2: In-context learning pattern](../zh/images/day20/in-context-learning-pattern.png)
*Caption: Few-shot prompting works by giving the model examples that reveal the hidden pattern, output schema, and task boundary.*

This matters because prompts can provide more than instructions. They can provide **implicit supervision**. A few carefully chosen examples often communicate:

- the desired output format,
- what counts as relevant evidence,
- how detailed the answer should be,
- edge cases to handle,
- and what kind of errors are unacceptable.

A good analogy is showing someone three solved equations before giving them a fourth. You are not retraining their brain. You are helping them infer the rule of the game.

Why does few-shot prompting often outperform a purely verbal description? Because examples are concrete. Natural language instructions can be ambiguous. Examples pin down the exact mapping.

That said, examples consume context window and can anchor the model too strongly. Bad examples teach the wrong lesson. Prompt engineering is therefore partly about choosing whether the task is better specified by **rules**, **examples**, or both.

---

## 4. Core prompt patterns that actually matter

**One-sentence summary**: Most useful prompt engineering can be reduced to a small set of reliable patterns rather than a bag of hacks.

### 4.1 Instruction-first prompting

Put the task early and state it plainly. Modern instruction-tuned models respond better when the main instruction appears before noisy context.

Bad:

> Here are logs, some notes, a long story, and by the way summarize the incident.

Better:

> Summarize the incident in 5 bullets, focusing on user impact, root cause, and mitigation. Use the logs below.

Instruction-first prompting helps because the model forms a task frame early. Later context is then interpreted through that frame.

### 4.2 Delimited context

Separate instructions from evidence using clear markers such as XML tags, markdown code fences, or section headers.

Example:

```text
Task: Answer only from the provided document.

<Document>
...
</Document>
```

Delimiters reduce accidental blending of instructions and data. They also make it easier for the model to learn where quoted material starts and ends.

### 4.3 Output schema prompting

If you need machine-readable output, specify the schema explicitly.

```json
{
  "severity": "low|medium|high",
  "root_cause": "string",
  "actions": ["string"]
}
```

This usually works better than saying, "be structured." Models handle explicit constraints far better than implied ones.

### 4.4 Few-shot prompting

Add examples when verbal instructions are insufficient, especially for classification, transformation, extraction, and style imitation tasks.

### 4.5 Decomposition

Break a hard task into smaller stages. Instead of one giant prompt that asks for analysis, verification, formatting, and prioritization all at once, split the work.

![Figure 3: Prompt decomposition tree](../zh/images/day20/prompt-decomposition-tree.png)
*Caption: Decomposition helps when a single instruction is too underspecified. Smaller subtasks reduce ambiguity and make checking easier.*

This is one reason agent systems exist. A strong agent is often just prompt engineering plus tools plus state management plus validation.

### 4.6 Ask for checks, not just answers

If the task has objective constraints, ask the model to verify them.

For example:

- "Before finalizing, check whether every claim is supported by the provided text."
- "Return INVALID if required fields are missing."
- "List assumptions separately from facts."

This does not make the model perfectly reliable, but it improves failure visibility.

---

## 5. Chain-of-thought, hidden reasoning, and why prompting can improve reasoning

**One-sentence summary**: Some prompts improve reasoning not by adding intelligence, but by making the model spend its computation in a more useful pattern.

A famous example is **Chain-of-Thought (CoT)** prompting: asking the model to reason step by step. On many tasks, especially math and symbolic reasoning, performance improves.

Why? One intuition is that the prompt encourages the model to generate intermediate tokens that act like scratch work. Those tokens create additional context for later steps. Instead of jumping directly from question to answer, the model builds a temporary reasoning trace.

If we write the answer probability as depending on latent intermediate states $z$,

$$
P(y \mid x) = \sum_z P(y \mid z, x) P(z \mid x),
$$

then chain-of-thought prompting can be viewed as making some useful intermediate $z$ explicit in text rather than leaving them implicit inside one forward-pass continuation.

But there are caveats.

- Not every task benefits from longer reasoning traces.
- Verbose reasoning can introduce new errors.
- Exposed chain-of-thought may be undesirable for privacy, safety, or product reasons.
- Some modern systems get the benefit through hidden reasoning or multi-step internal scaffolding rather than showing every step to the user.

So the deeper lesson is not "always ask for step-by-step reasoning." It is: **good prompts allocate computation and structure to the hard parts of the task**.

---

## 6. Prompt engineering versus fine-tuning

**One-sentence summary**: Prompting is flexible and cheap at inference time, while fine-tuning moves repeated behavior into the model weights.

When should you keep improving prompts, and when should you fine-tune?

Prompt engineering is strongest when:

- the task changes often,
- you need quick iteration,
- you can provide task-specific context at runtime,
- or the behavior is mostly about formatting, policy, or routing.

Fine-tuning is stronger when:

- the desired behavior is repeated at large scale,
- the task needs stable behavior across many inputs,
- examples are too long to keep repeating in the prompt,
- or you want the model to internalize a narrower distribution.

A rough trade-off is:

$$
\text{Total cost} \approx \text{prompt tokens per request} \times N \quad \text{vs.} \quad \text{one-time tuning cost} + \text{smaller runtime prompts}.
$$

If $N$ is huge, the runtime cost of long prompts can become painful. This is why production systems often start with prompt engineering and only later move stable behavior into fine-tuning.

Prompt engineering is best seen as the fastest control surface. Fine-tuning is the heavier structural change.

---

## 7. Common prompt failures and why they happen

**One-sentence summary**: Prompt failures are usually failures of specification, salience, overload, or verification.

### 7.1 Vague goals

"Analyze this" is underspecified. Analyze for what? Risk? sentiment? timeline? root cause? audience level?

### 7.2 Conflicting instructions

If you ask for "be concise" and also "cover every detail," the model must guess which instruction matters more.

### 7.3 Context overload

More context is not always better. Irrelevant content can drown the signal.

![Figure 4: Specificity versus performance](../zh/images/day20/specificity-vs-performance.png)
*Caption: Better prompts usually add useful structure first. After a point, too much detail or irrelevant context can hurt quality and increase latency.*

A nice rule is: **add information that changes the correct answer, remove information that only increases confusion**.

### 7.4 Missing format constraints

If you need a table, schema, or ranked list, ask for it explicitly.

### 7.5 No grounding requirement

If the model should only use the provided documents, say so. Otherwise it may combine prompt evidence with background knowledge and produce confident but unsupported claims.

### 7.6 No validation loop

If the task is high stakes, one-pass prompting is rarely enough. Add checks, tool use, or a second verification step.

This is a major shift in modern LLM application design. People used to hunt for one perfect prompt. Strong systems now use **prompt pipelines** with retrieval, tools, self-checks, and structured outputs.

---

## 8. A practical recipe for prompt design

**One-sentence summary**: Start from the desired output and failure modes, then work backward to the minimum prompt structure needed.

Here is a practical workflow that works better than prompt superstition.

### Step 1: Define success precisely

What should a correct answer look like? What wrong answers are common but unacceptable?

### Step 2: Specify the output format

If downstream code or humans need a particular structure, lock that down early.

### Step 3: Add only the necessary context

Include the evidence, examples, or metadata that truly changes the answer.

### Step 4: Add constraints

Examples: "cite the source sentence," "do not use external knowledge," "if uncertain, say insufficient information."

### Step 5: Add examples if ambiguity remains

Few-shot examples are especially useful when there is a subtle decision boundary.

### Step 6: Evaluate on adversarial cases

Do not test only easy inputs. Test edge cases, conflicting evidence, missing fields, and noisy documents.

### Step 7: Decide whether the problem is really a prompt problem

Sometimes people keep rewriting prompts when the real issue is missing retrieval, missing tools, bad data, or a model that is too weak for the task.

Prompt engineering is powerful, but it is not a miracle patch for every product failure.

---

## 9. Common misconceptions

**One-sentence summary**: Many myths about prompt engineering come from treating prompts as magic spells instead of temporary task specifications.

### Misconception 1: There exists one perfect universal prompt

No. Prompts are highly task-dependent and model-dependent.

### Misconception 2: Longer prompts are always better

No. Useful information helps. Irrelevant verbosity hurts.

### Misconception 3: Prompt engineering will disappear because models get smarter

Partly false. Crude wording sensitivity may decline, but specification will always matter. Even a brilliant assistant still needs clear instructions, good context, and explicit constraints.

### Misconception 4: Prompt engineering is only about wording tricks

Also false. The deeper work is task design: examples, structure, tools, retrieval, validation, and output contracts.

---

## Optional Math: why examples can shift decisions

For a simple classification-like prompt, the model chooses

$$
\hat{y} = \arg\max_y P(y \mid x).
$$

Few-shot examples change $x$, which can change the ranking among candidate labels. If examples make the intended mapping clearer, the margin

$$
\Delta = \log P(y^* \mid x) - \log P(y' \mid x)
$$

for the correct label $y^*$ over a wrong label $y'$ may increase. In practical terms, the model becomes more confident in the decision boundary the prompt is teaching.

This is not gradient descent during inference. It is conditional computation over a richer context.

---

## Further Reading

- Brown et al., *Language Models are Few-Shot Learners* (2020)
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2022)
- Reynolds and McDonell, *Prompt Programming for Large Language Models* (2021)
- Anthropic and OpenAI prompt guides for practical instruction design

---

## Reflection

If you had to improve an LLM application and were allowed to change only one thing, would you change the prompt, the retrieved context, the output schema, or the validation loop? Your answer says a lot about what you think the real bottleneck is.

---

## Takeaway

Prompt engineering is best understood as **interface design for model cognition**. You are shaping the temporary information environment in which the model reasons. The best prompts are not clever because they use secret phrases. They are good because they make the task legible: clear goal, relevant context, explicit constraints, and verifiable output.

That is also why prompt engineering connects directly to the next stage of modern systems. Once one prompt is not enough, we start building retrieval, tool use, memory, and agents around the model. In that sense, prompt engineering is not a side trick. It is the first layer of LLM systems engineering.
