# Day 16: 采样策略

> **核心问题**：如果语言模型已经给出了下一个 token 的概率分布，我们该怎样把这个分布变成既有用、又连贯、还不至于无聊的文本？

---

## 开场

语言模型并不是直接“写出”下一个词。它真正做的是给出一张候选菜单，列出若干可能的下一个 token，以及各自的概率。解码（decoding）或采样策略（sampling strategy），决定了我们如何从这张菜单里做选择。很多初学者把它当成实现细节，但它其实非常关键。模型听起来是稳健、呆板、还是胡言乱语，往往就差在这里。

你可以把模型想成一个会即兴演奏的爵士钢琴手，它知道很多“都说得过去”的下一拍。采样策略像乐队指挥。很保守的指挥说：“永远弹最可能的那个音。” 结果通常整齐，但也很死板。很放飞的指挥说：“随便来。” 结果可能有惊喜，也可能直接跑调。好的生成效果，往往就在两者之间。你需要足够的自由度，才能有变化；也需要足够的约束，才能维持结构。

这就是为什么同一个基础模型，在不同解码设置下会像完全不同的系统。Temperature、top-k、top-p 不是装饰性参数，它们控制的是“不确定性如何被表达出来”。Beam search 也不只是“更努力地搜一搜”，因为在开放式文本生成里，概率最高的序列，并不一定最像人写的，也不一定最值得读。

> **什么是 beam search？** 不像普通解码每步只选一个最可能的 token（贪心解码），beam search 同时保留前 $B$ 条候选序列，最后选总分最高的。比贪心更全面，但倾向于生成安全、通用的文本。我们会在第 4 节详细解释。

---

## 1. 解码问题，出现在模型完成“本职工作”之后

自回归语言模型（autoregressive language model）的核心任务可以写成：

$$
\begin{aligned}
P(x) &= \prod_{t=1}^{T} P(x_t \mid x_{<t}) \\
\hat{x}_{t+1} &\sim P(\cdot \mid x_{\le t})
\end{aligned}
$$

第一行表示，一个序列的概率可以分解成逐步的条件概率乘积。第二行里真正藏着整个解码问题的关键，也就是符号 $\sim$。模型告诉你“哪些 token 更可能”，但并没有替你决定“到底选哪一个”。

这个选择很重要，因为真实的语言分布通常并不只有一个合理答案。比如在 “The capital of France is” 这种上下文里，分布会非常尖锐，几乎没什么悬念。但在 “In the moonlit alley, she whispered” 这种上下文里，合理的续写会很多。解码器必须决定，到底是偏向最有把握的候选，还是保留一定探索空间。

![Figure 1: Temperature reshapes the same logits](./images/day16/temperature-distribution.png)
*图 1：Temperature 不会改变模型原始 logits，而是改变 logits 转成概率分布后的尖锐程度。于是同一个模型会表现得更保守，或者更大胆。*

一个很有用的心智模型是：模型给出的是**beliefs（信念）**，解码算法把这些信念变成**behavior（行为）**。训练决定模型“知道什么”，采样决定模型“怎么说出来”。

### 1.1 从 logits 到概率

在采样前，模型输出的是 **logits**，也就是每个词表 token 的未归一化分数。我们通过 softmax 把它们变成概率：

$$
\begin{aligned}
p_i &= \frac{\exp(z_i)}{\sum_j \exp(z_j)} \\
\sum_i p_i &= 1
\end{aligned}
$$

因为 softmax 带指数，所以 logits 的微小变化，可能导致概率的明显变化。这也是为什么解码参数往往“手一拧，效果就完全不一样”。

### 1.2 为什么贪心解码常常不够好

最简单的策略是 **贪心解码（greedy decoding）**，每一步都选当前概率最高的 token。它计算便宜，也很稳定，但在开放式生成里经常不理想。

原因在于，局部最优会逐步累积。如果模型每一步都选“最安全”的词，长文本最终就容易滑向平淡、重复、模板化的表达。你可以把它想成在城市里永远选择最大、最亮、最好走的路。你当然不太会迷路，但也很难看到任何有趣的东西。

不过，贪心解码并不是一无是处。在信息抽取、结构化格式输出、低随机性的工具调用等任务里，它依然很有价值。

---

## 2. Temperature，最简单也最常被误解的参数

Temperature 的作用是先缩放 logits，再做 softmax：

$$
\begin{aligned}
p_i^{(T)} &= \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
\end{aligned}
$$

这个公式可以直接读出结论：

- 当 $T < 1$ 时，分布更尖锐。
- 当 $T = 1$ 时，分布不变。
- 当 $T > 1$ 时，分布更平坦。
- 当 $T \to 0$ 时，行为接近贪心解码。
- 当 $T \to \infty$ 时，分布趋近于均匀随机。

Temperature **不会给模型增加知识**，它改变的是风险偏好。低 temperature 的含义是：“更强地相信模型当前最看好的选项。” 高 temperature 的含义是：“给排位更靠后的选项更多出场机会。”

### 2.1 什么时候适合低 temperature

低 temperature 通常适合：

- 事实型问答，
- 对格式要求严格的输出，
- 希望减少奇怪尾部 token 的场景，
- 希望复现性更高的系统。

### 2.2 什么时候适合高 temperature

高 temperature 可能适合：

- 头脑风暴，
- 多版本草稿生成，
- 更有变化的对话，
- 帮助模型跳出重复局部循环。

但代价很明确，高 temperature 提升多样性，是用可靠性换来的。如果原始分布尾部就已经混着很多质量差的候选，那么把分布拉平，只会让这些差候选更容易被采到。

### 2.3 只调 temperature 往往不够

一个常见误解是，以为 temperature 就是唯一重要的随机性控制参数。其实不是。Temperature 会整体改造分布，包括糟糕的尾部。实战里大家常常把 temperature 和截断策略一起用，比如 top-k 或 top-p。

---

## 3. Top-k、top-p，与“截掉尾巴”这件事

**这一节换一个控制杠杆来看问题：不是把概率重新捏形，而是直接取消一部分候选的抽样资格。**

如果说 temperature 改变的是分布的形状，那么截断（truncation）改变的就是分布的**支持集（support）**，也就是哪些 token 还有资格被采样。

### 3.1 Top-k 采样

**Top-k sampling** 的做法是，只保留当前概率最高的前 $k$ 个 token，把它们重新归一化，再从中采样：

$$
\begin{aligned}
S_k &= \text{按概率排序后前 } k \text{ 个 token 的集合} \\
p_i^{(k)} &= \begin{cases}
\frac{p_i}{\sum_{j \in S_k} p_j} & i \in S_k \\
0 & i \notin S_k
\end{cases}
\end{aligned}
$$

它的优点很直观：直接砍掉低概率尾部，避免从 5 万词表里抽到一些几乎不该出现的垃圾 token。

但它的问题也很明显，合适的 $k$ 其实依赖上下文。有的上下文非常确定，合理候选只有几个；有的上下文更模糊，合理候选可能很多。固定的 $k$ 不会自动适应这些差异。

### 3.2 Top-p，也叫 nucleus sampling

**Top-p sampling** 想解决的正是这个问题。它不是固定保留多少个 token，而是保留“累计概率刚好达到阈值 $p$ 的最小集合”。比如 $p=0.9$ 或 $0.95$。

![Figure 2: Top-k uses a fixed count, top-p uses a probability mass target](./images/day16/top-k-vs-top-p.png)
*图 2：Top-k 总是保留固定数量的候选，top-p 则根据当前分布的集中程度，动态决定候选集合大小。*

形式化写法是，把 token 按概率从大到小排序，选择最小集合 $S_p$，满足：

$$
\begin{aligned}
\sum_{i \in S_p} p_i \ge p
\end{aligned}
$$

这也是为什么 top-p 在实际系统中非常流行。模型很有把握时，nucleus 只会包含少数几个 token；模型更犹豫时，候选集合就会自动扩大。它比固定宽度的 top-k 更自适应。

### 3.3 为什么 top-p 经常比 top-k 更稳

真实语言分布不是静态的。有些位置概率尖锐，有些位置很平。固定 top-k 把这两种情况一视同仁，而 top-p 则更像在说：“保留大多数概率质量，尾巴就别碰了。” 这通常能得到更好的质量与多样性平衡。

### 3.4 Temperature 和 top-p 经常一起出现

典型的生产配置往往是：

1. 先用 temperature 调整分布的尖锐程度，
2. 再用 top-p 去掉低质量尾部，
3. 最后在保留下来的 nucleus 里采样。

这种组合很好理解。Temperature 控制“敢不敢放开”，top-p 控制“就算放开，也别太离谱”。

---

## 4. Beam Search，为什么翻译里常见，聊天里却常翻车

**这一节讲的是另一类解码方法，不是再加一个普通参数。**

**Beam Search（束搜索）** 从严格意义上说，不太像常规采样。它会在每一步保留前 $B$ 条得分最高的部分序列，继续展开，再筛选。这比贪心解码更接近全局搜索。

> **Beam search 怎么工作（具体例子，beam width = 3）：**
>
> 普通解码（贪心）每步只选概率最高的那一个 token——就像走迷宫时永远选眼前最近的路。快，但可能错过更好的路线。
>
> Beam search 同时保留 B 条候选路径：
>
> **第一步：** 模型输出候选 token："The" (0.5), "A" (0.3), "This" (0.15)... 保留前 3 条：`["The", "A", "This"]`
>
> **第二步：** 对每条路径分别扩展：
> - "The cat" (0.5 × 0.4 = 0.20)
> - "The dog" (0.5 × 0.3 = 0.15)
> - "A cat" (0.3 × 0.4 = 0.12)
> - "A dog" (0.3 × 0.3 = 0.09)
> - "This is" (0.15 × 0.5 = 0.075)
>
> 按总分保留前 3 条：`["The cat", "The dog", "A cat"]`
>
> **重复**直到生成结束，最后选总分最高的完整序列。

在机器翻译、语音识别这种“存在比较明确目标序列”的任务中，beam search 往往很有效。因为在这些任务里，序列概率高，通常也更接近高质量答案。

但开放式文本生成不是这样。对话、故事、长文创作里，最高概率的续写经常意味着最泛化、最安全、最无聊的表达。这就是所谓的 **likelihood trap（似然陷阱）**。模型很擅长给“大家都不会反对”的句子高分，但人类真正喜欢的文字，未必就是这种统计均值。

所以，beam search 不是过时了，而是它更适合那些“高概率序列≈高质量答案”的任务。不要因为它搜索更全面，就默认它对所有生成任务都更好。

---

## 5. 真实系统里，解码通常是“组合拳”

**一个更干净的心智模型是：先选“基础解码家族”，再叠加那些可以组合使用的控制项。**

现实中的生成系统，很少只依赖一个参数。更常见的是一整套组合。

![Figure 3: Decoder families vs stackable controls](./images/day16/sampling-strategy-map.png)
*图 3：Beam Search 更适合被看成另一种解码家族。Temperature、top-k、top-p、重复惩罚、停止规则等，则更像采样式解码上可以叠加的控制项。*

一个常见配置可能是：

- temperature = 0.7，
- top-p = 0.9，
- 再加 frequency penalty 或 repetition penalty，
- 再加最大生成长度和停止条件。

### 5.1 重复惩罚为什么重要

采样策略不仅关乎“创造力”，也关乎“别陷入循环”。语言模型有时会进入重复模式，因为一旦某个短语开始重复，它就会在后续上下文里不断强化自己。重复惩罚通过降低已出现 token 的概率，来打断这种自我强化。

这在长文本生成里特别重要。否则就算基础模型不错，也可能像卡住的录音机一样反复念同一句话。

### 5.2 Typical sampling 和 min-p

近年来又出现了一些改进型截断方法：

- **Typical sampling**：更偏好“惊讶度接近平均值”的 token，而不是单纯盯着概率最高的那几个。
- **Min-p sampling**：保留那些相对 top token 概率不太低的候选。

这里也顺手把几个缩略词讲清楚。**Top-k** 的意思是“固定保留 k 个候选”，**top-p** 也叫 **nucleus sampling（核采样）**，意思是“保留足够多的候选，使累计概率达到阈值 p”，而 **Beam Search** 的意思是“并行保留多条高分部分序列继续扩展”。它们相关，但不是同一种控制旋钮。

你不需要死记所有变体。更重要的是看懂背后的统一问题：**我们怎样既保留有意义的不确定性，又尽量过滤低质量尾部？**

---

## 6. 代码示例：把主要策略实现一遍

```python
import torch
import torch.nn.functional as F


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    从模型给出的 logits 中采样一个下一个 token。

    参数说明：
        logits: 下一步词表 logits，一维张量。
        temperature: 控制分布尖锐程度，越低越保守。
        top_k: 只保留概率最高的前 k 个 token。
        top_p: 只保留累计概率达到 p 的最小集合。
    """
    # 避免除零。极小的 temperature 会逼近贪心解码。
    temperature = max(temperature, 1e-5)
    scaled_logits = logits / temperature

    # 先把 logits 变成概率。
    probs = F.softmax(scaled_logits, dim=-1)

    if top_k is not None:
        # 找到前 k 个 token，并把其他 token 概率清零。
        top_values, top_indices = torch.topk(probs, k=top_k)
        filtered = torch.zeros_like(probs)
        filtered[top_indices] = top_values
        probs = filtered / filtered.sum()

    if top_p is not None:
        # 先按概率从大到小排序，再截取 nucleus。
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # 要保留“刚刚把累计概率推过阈值”的那个边界 token，
        # 否则保留下来的概率质量会小于 top_p 的定义。
        keep_mask = cumulative <= top_p
        first_cross = torch.nonzero(cumulative > top_p, as_tuple=False)
        if len(first_cross) > 0:
            keep_mask[first_cross[0].item()] = True
        keep_mask[0] = True  # 至少保留概率最高的 token。

        filtered = torch.zeros_like(probs)
        filtered[sorted_indices[keep_mask]] = probs[sorted_indices[keep_mask]]
        probs = filtered / filtered.sum()

    # 按最终概率分布随机采样一个 token。
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item(), probs
```

这段代码虽然短，但已经把解码的核心思路暴露出来了。所有主流策略，本质上都在修改两件事之一：要么改分布形状，要么改哪些 token 还能参与抽样。Top-p 里最容易写错的地方，就是那个“边界 token”也必须保留，否则实现会和 nucleus sampling 的定义不一致。

---

## 7. 常见误解

### ❌ “Temperature 越低，事实性就一定越高”

不一定。低 temperature 只是减少随机性，不会修复模型内部本来就错误的知识。如果模型最相信的答案就是错的，那么 temperature = 0 也只会更坚定地输出这个错误。

### ❌ “Beam search 搜得更多，所以一定更好”

只有当“模型概率”和“人类质量判断”高度一致时，这种说法才成立。在开放式文本生成里，这两者往往并不一致。

### ❌ “Top-k 和 top-p 差不多，可以随便替换”

它们确实都属于截断策略，但机制不同。Top-k 固定候选数量，top-p 固定概率质量，因此在不同上下文下会表现出明显差异。

### ❌ “采样只适用于创意写作”

不是。结构化抽取、工具调用、摘要、推理任务，同样都需要根据任务特性选择合适的解码方式。

### ❌ “输出质量差，就把 temperature 调高”

这经常是错方向。如果模型本来就不确定，或者尾部分布很脏，提高 temperature 只会进一步放大噪声。很多糟糕输出，根源其实在 prompt、模型能力或外部知识缺失，而不是单一采样参数。

---

## 8. 实战中怎么选

**真正的实战问题不是“哪个解码最好”，而是“这个任务能容忍多大不确定性”。**

下面这组经验规则通常很有效：

### 对事实问答、抽取、工具调用
- temperature 先从 0.0 到 0.3 开始，
- top-p 可以接近 1.0，甚至关闭，
- 优先保证稳定性和格式正确率。

### 对一般聊天
- temperature 可以从 0.6 到 0.8 开始，
- top-p 常见设为 0.9 到 0.95，
- 如果输出很长，建议配合重复惩罚。

### 对头脑风暴或创意写作
- temperature 可以尝试 0.8 到 1.1，
- top-p 依然可保持在 0.9 左右，
- 更推荐一次生成多个候选，再做后排序。

### 对翻译或受约束生成
- 贪心解码或 beam search 可能是合理起点，
- 但最好总是和采样基线做比较，而不是默认更复杂的搜索就更优。

更深一层的原则其实很简单：**让解码策略匹配任务对不确定性的容忍度。** 有的任务惩罚变化，有的任务则需要变化。

---

## 9. 延伸阅读

### 入门
1. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)  
   经典论文，解释为什么贪心解码和 beam search 容易生成退化文本，也推动了 top-p 的流行。
2. [How to Generate Text: Using Different Decoding Methods for Language Generation with Transformers](https://huggingface.co/blog/how-to-generate)  
   Hugging Face 的实战型教程，适合快速建立整体概念。

### 进阶
1. [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191)  
   从理论角度解释为什么截断式采样常常有效。
2. [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)  
   介绍 typical sampling 这种替代 top-k/top-p 的思路。

### 论文
1. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
2. [Language Modeling with Nucleus Sampling](https://openreview.net/forum?id=rygGQyrFvH)
3. [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)

---

## 反思问题

1. 为什么“每一步都选概率最高的 token”经常会让长文本质量变差，而不是变好？
2. 哪些任务里你宁愿牺牲一点自然感，也要换取完全确定性的输出？为什么？
3. 如果一个模型在 temperature = 0 时仍然会幻觉，这说明问题根源更可能出在哪里？

---

## 总结

| 概念 | 一句话解释 |
|------|------------|
| 贪心解码 | 每一步都选最可能的 token，稳定但常常无聊 |
| Temperature | 重塑分布的尖锐程度，让模型更保守或更随机 |
| Top-k | 只从前 k 个最可能 token 中采样 |
| Top-p | 只从累计概率达到阈值的最小集合中采样 |
| Beam search | 搜索高概率序列，适合受约束任务，但对开放式生成常过于保守 |

**核心结论**：语言模型输出的是概率，不是最终文本。采样策略就是从“模型的不确定性”到“用户看到的行为”的桥梁。好的解码，不是盲目追求随机，也不是盲目追求确定，而是有纪律地管理不确定性。

---

*Day 16 of 60 | LLM Fundamentals*  
*Word count: ~1926 Chinese characters equivalent | Reading time: ~16 minutes*
