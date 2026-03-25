# Day 3: 注意力的诞生

> **核心问题**：注意力机制解决了什么问题，这个机制究竟是如何工作的？

---

## 引言

2014 年，神经机器翻译令人兴奋但也令人沮丧。Seq2seq 模型可以翻译句子，但有一个致命缺陷：无论输入句子有多长，所有内容都必须压缩到一个固定大小的向量中。

想象一下：让你把一份 100 词的法律文件翻译成法语，但你只能用一张便利贴做笔记。这就是 seq2seq 模型当时的处境。

瓶颈是显而易见的：长句子表现很差。信息丢失了。解码器必须从一个同时包含所有其他内容的压缩 blob 中提取 "European Economic Area agreement"。

然后，Bahdanau、Cho 和 Bengio 在 2014 年发表了论文："Neural Machine Translation by Jointly Learning to Align and Translate"。关键洞察简单但深刻：**不是把所有内容压缩到一个向量（encoder 最后一个隐藏状态）中，而是让解码器回看所有编码器状态，并决定哪些与每个输出词相关。**

这就是注意力（Attention）的诞生。

![有无注意力的 Seq2seq 对比](./images/day03/seq2seq-attention-comparison.png)
*图 1：左：没有注意力时，所有信息必须通过单一的"瓶颈"向量。右：有了注意力，解码器可以直接访问所有编码器隐藏状态，学习到的权重指示相关性。*

三年后，"Attention Is All You Need" 更进一步：如果我们围绕注意力构建一个完整的架构，完全不用循环呢？Transformer 诞生了，随之而来的是 GPT、BERT 和所有现代大语言模型的基础。

今天，我们将从第一性原理理解注意力——它为什么被发明，它在数学上究竟如何工作，以及为什么"缩放点积"公式成为了标准。

---

## 1. 瓶颈问题

### 1.1 Seq2seq 回顾

让我们回顾一下 seq2seq 模型如何工作（Day 2 中讨论过）：

1. **编码器（Encoder）**：逐词处理输入序列，产生隐藏状态 h₁, h₂, ..., hₙ
2. **上下文向量（Context Vector）**：取最终隐藏状态 hₙ 作为整个输入的"摘要"
3. **解码器（Decoder）**：基于这个上下文向量生成输出序列

问题在于第 2 步。最终隐藏状态必须编码*所有内容*：主语、宾语、动词时态、词与词之间的关系——全部压缩到一个 512 或 1024 维的向量中。

这对短句子有效。对长句子呢？信息就丢失了。

### 1.2 实验证据

Bahdanau 等人用实验证明了这一点。他们绘制了 BLEU 分数（翻译质量）与句子长度的关系：

| 句子长度 | 标准 Seq2seq | 带注意力 |
|---------|-------------|---------|
| 10-20 词 | 好 | 好 |
| 30-40 词 | 下降 | 仍然好 |
| 50+ 词 | 差 | 保持稳定 |

标准模型在长句子上性能崩溃。基于注意力的模型无论长度如何都保持质量。

### 1.3 关键洞察

人类译者不会读完整个源句子，把它压缩到脑子里，然后写翻译。他们来回看。翻译 "European Economic Area" 时，他们会看那几个特定的英文词，而不是开头的 "The"。

注意力赋予神经网络这种能力：**动态的、基于内容的**对源序列的查找。

---

## 2. 注意力如何工作

### 2.1 核心机制

注意力计算值的加权和，其中权重由查询与一组键之间的相似度决定。

**三个关键概念：**
- **查询（Query，Q）**：我在找什么？
- **键（Keys，K）**：我有什么可用的？
- **值（Values，V）**：我要检索什么？

> **图书馆比喻**：想象你在图书馆找书。
> - **Query** = 你的搜索描述："我要找一本关于机器学习的书"
> - **Keys** = 书架上每本书的标签/索引："ML入门"、"深度学习"、"烹饪"...
> - **Values** = 书里面的实际内容
> 
> Keys 告诉你**有哪些选项**（available/可用）。Values 是你**实际拿到的东西**（retrieve/检索）。
> 为什么要分开？因为搜索条件和返回内容可以不一样！

在原始的 seq2seq 注意力中：
- Query = 当前解码器隐藏状态（我需要生成什么）
- Keys = 所有编码器隐藏状态（我可以看什么）
- Values = 与 Keys 相同（我要检索什么）

### 2.2 三个步骤

![注意力计算步骤](./images/day03/attention-computation-steps.png)
*图 2：注意力的三个步骤：(1) 计算查询与键之间的相似度分数，(2) 通过 softmax 归一化分数得到权重，(3) 计算值的加权和。*

**步骤 1：计算分数**

对于每个键 kᵢ，计算与查询 q 的相似度分数：

```
score_i = similarity(q, k_i)
```

常见的相似度函数：
- **点积（Dot product）**：`score = q · k`（快速、简单）
- **缩放点积（Scaled dot product）**：`score = (q · k) / √d_k`（Transformer 使用的）
- **加性（Additive）**：`score = v^T · tanh(W_q q + W_k k)`（原始 Bahdanau）

![相似度函数对比](./images/day03/similarity-functions-comparison.png)
*图：三种相似度函数的对比。点积简单但高维时分数会爆炸。缩放点积通过除以 √d_k 解决这个问题。加性表达能力更强但更慢。*

> **为什么点积会"爆炸"？**
> 
> **抛硬币类比**：想象抛硬币，每枚结果是 +1 或 -1：
> 
> | 硬币数 | 可能的总和 | 范围 |
> |--------|-----------|------|
> | 1 枚 | -1, +1 | ±1 |
> | 3 枚 | -3, -1, +1, +3 | ±3 |
> | 100 枚 | -100 到 +100 | ±100 |
> 
> 硬币越多 = 越多"波动源"叠加 = 结果的范围越大。
> 
> **点积也是一样：**
> $$q \cdot k = \underbrace{q_1 k_1}_{波动源1} + \underbrace{q_2 k_2}_{波动源2} + ... + \underbrace{q_d k_d}_{波动源d}$$
> 
> 每个 qᵢkᵢ 是一个方差为 1 的"波动源"。d 个波动源相加 → 总方差 = d。
> 
> **维度越高，点积的值越可能"跑很远"！**
> 
> | 维度 d | 分数的标准差 | 典型范围 |
> |--------|-------------|----------|
> | 1 | 1 | -3 ~ +3 |
> | 64 | √64 = 8 | -24 ~ +24 |
> | 512 | √512 ≈ 22.6 | -68 ~ +68 |
> 
> 分数太大 → softmax 变成近似 one-hot → 梯度消失 → 训练失败。
> 
> **什么是 "one-hot"？** 只有一个位置是 1，其他全是 0 的向量。
> 
> **为什么大分数会导致 one-hot？** 因为 softmax 用了**指数函数**，指数函数对大数值极度敏感：
> 
> | x | e^x |
> |---|-----|
> | 1 | 2.7 |
> | 10 | 22,026 |
> | 20 | 485,165,195 |
> | 30 | 10,686,474,581,524 |
> 
> 差距 10 → 指数差距 **2 万倍**！
> 
> **小分数** → `softmax([1, 2, 3])`：
> ```
> e^1 = 2.7,  e^2 = 7.4,  e^3 = 20.1
> 总和 = 30.2
> → [2.7/30.2, 7.4/30.2, 20.1/30.2]
> → [0.09, 0.24, 0.67]  ✅ 平滑分布
> ```
> 
> **大分数** → `softmax([10, 20, 30])`：
> ```
> e^10 = 22,026
> e^20 = 485,165,195
> e^30 = 10,686,474,581,524  ← 碾压前两个！
> 总和 ≈ e^30
> → [≈0, ≈0, ≈1]  ❌ one-hot
> ```
> 
> 指数函数让大的更大、小的更小。当分数差距大时，最大值"吃掉"其他所有值 → one-hot。
> 
> 当 softmax 输出接近 0 或 1 时，梯度接近 0 → 参数停止更新 → 训练失败。
> 
> **解决方案**：除以 √d_k，把方差归一化回 1。

**步骤 2：通过 Softmax 归一化**

将分数转换为概率分布（权重和为 1）：

```
α_i = softmax(score_i) = exp(score_i) / Σ_j exp(score_j)
```

现在 αᵢ ∈ [0, 1] 且 Σαᵢ = 1。这些就是**注意力权重**。

> **关键区分：**
> - **Score** = scaled 点积（原始相似度）
> - **Attention Weight** = softmax(score)（归一化后的关注程度）

**步骤 3：加权和**

将上下文向量计算为值的加权组合：

```
context = Σ_i α_i · v_i
```

就这样！上下文向量是所有值的加权平均，其中权重指示对每个位置"给予多少注意力"。

> **静态 vs 动态 Context**
> 
> | 模型 | Context Vector | 特性 |
> |------|----------------|------|
> | **原始 Seq2Seq** | 固定的（encoder 最后一个隐藏状态） | **静态**，所有 decoder 步骤都一样 |
> | **Attention** | 基于 Query 的动态加权和 | **动态**，每一步都不同 |
> 
> 例子：翻译 "I love cats" → "我 爱 猫"
> ```
> 生成 "我" → context 重点关注 "I"    (权重: I=0.8, love=0.1, cats=0.1)
> 生成 "爱" → context 重点关注 "love" (权重: I=0.1, love=0.8, cats=0.1)
> 生成 "猫" → context 重点关注 "cats" (权重: I=0.1, love=0.1, cats=0.8)
> ```
> 
> **Attention 让 context vector 从"一张固定照片"变成了"可以随时对焦的摄像头"** 📷

### 2.3 直觉：软寻址

把注意力想象成**软的、可微分的内存查找**。

在传统数据库中：
```
query = "European Economic Area"
result = exact_match(query, database)  # 返回一条记录或空
```

在注意力中：
```
query = decoder_state
weights = softmax(similarity(query, all_keys))  # 返回所有记录的权重
result = weighted_sum(weights, all_values)  # 所有记录的软混合
```

"软"的部分至关重要：不是硬选择，而是一个可微分的操作，可以用反向传播端到端训练。

---

## 3. 注意力可视化：对齐

注意力有一个美妙的特性：权重是可解释的。它们显示了模型在生成每个目标词时"看向"哪些源词。

![注意力热图](./images/day03/attention-heatmap.png)
*图 3：英语→法语翻译的注意力权重。每一行显示模型在生成那个法语词时关注什么。大致的对角线表示词对齐。*

注意：
- "L'" 强烈关注 "The"（冠词对齐）
- "accord" 关注 "agreement"（直接翻译）
- "européenne" 关注 "European"（形容词，在法语中位置不同）
- "zone" 关注 "Area"（词序重排被优雅处理）

这是**学习到的对齐**，不是手工编码的规则。模型从翻译示例中发现，在生成法语词 X 时，应该看英语词 Y。

---

## 4. 缩放点积注意力

Transformer 论文（"Attention Is All You Need"）引入了一个特定的公式，成为了标准：**缩放点积注意力（Scaled Dot-Product Attention）**。

### 4.1 公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

让我们分解一下：

1. **QK^T**：点积矩阵。如果 Q 的形状是 (n, d_k)，K 的形状是 (m, d_k)，那么 QK^T 的形状是 (n, m)。第 (i, j) 个元素是查询 i 和键 j 之间的相似度。

2. **√d_k**：缩放因子。d_k 是键/查询的维度。

3. **softmax**：按行应用。每行和为 1。

4. **V**：值矩阵。形状 (m, d_v)。输出形状 (n, d_v)。

![缩放点积注意力](./images/day03/scaled-dot-product-attention.png)
*图 4：缩放点积注意力的数据流。通过 √d_k 缩放对稳定训练至关重要。*

### 4.2 为什么要除以 √d_k？

这是一个微妙但重要的细节。没有缩放，注意力在高维嵌入时会崩溃。

**问题**：

假设查询和键的元素是均值为 0、方差为 1 的独立随机变量。点积 q · k 是 d_k 个乘积的和：

$$
q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i
$$

每个乘积 qᵢ · kᵢ 的方差为 1（两个方差为 1 的变量的乘积）。和的方差为 d_k（独立变量的方差可加）。

所以点积的标准差是 √d_k。对于 d_k = 512，这大约是 22.6！

![缩放的重要性](./images/day03/scaling-importance.png)
*图 5：左：不同 d_k 值的点积分布——方差随维度增长。右：对 softmax 的影响——高方差导致尖峰分布，除了一个位置外所有位置的梯度都接近零。*

**为什么这很重要**：

当 softmax 输入有大的幅度时，输出变得极度尖峰——一个元素接近 1，其他接近 0。这导致：

1. **梯度消失**：极端值时 ∂softmax/∂x ≈ 0
2. **硬注意力**：没有平滑的学习信号，只有硬选择
3. **训练不稳定**：梯度要么爆炸要么消失

**解决方案**：

除以 √d_k 将方差归一化回 ~1：

```
scaled_score = (q · k) / √d_k
```

现在分数无论嵌入维度如何都有合理的幅度。Softmax 保持在梯度能正常流动的"有用"范围内。

> **快速检查**：如果 q, k 的维度 d_k = 64，未缩放的点积标准差 ≈ 8。缩放后：标准差 ≈ 1。对 softmax 好多了！

### 4.3 代码实现

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力。
    
    参数:
        Q: 查询，形状 (batch, n_queries, d_k)
        K: 键，形状 (batch, n_keys, d_k)
        V: 值，形状 (batch, n_keys, d_v)
        mask: 可选掩码，形状 (batch, n_queries, n_keys)
    
    返回:
        输出: 形状 (batch, n_queries, d_v)
        注意力权重: 形状 (batch, n_queries, n_keys)
    """
    d_k = Q.size(-1)
    
    # 步骤 1：计算缩放分数
    # (batch, n_queries, d_k) @ (batch, d_k, n_keys) -> (batch, n_queries, n_keys)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 可选：应用掩码（用于解码器自注意力、填充等）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 步骤 2：Softmax 得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 步骤 3：值的加权和
    # (batch, n_queries, n_keys) @ (batch, n_keys, d_v) -> (batch, n_queries, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


# 使用示例
batch_size = 2
n_queries = 4  # 例如 4 个解码器位置
n_keys = 6     # 例如 6 个编码器位置
d_k = 64
d_v = 64

Q = torch.randn(batch_size, n_queries, d_k)
K = torch.randn(batch_size, n_keys, d_k)
V = torch.randn(batch_size, n_keys, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"输出形状: {output.shape}")      # (2, 4, 64)
print(f"权重形状: {weights.shape}")     # (2, 4, 6)
print(f"每个查询的权重和: {weights.sum(dim=-1)}")  # 全是 1.0

# 可视化一个注意力模式
print(f"\n批次 0、查询 0 的注意力权重:")
print(weights[0, 0])  # 显示查询 0 如何关注所有 6 个键
```

输出：
```
输出形状: torch.Size([2, 4, 64])
权重形状: torch.Size([2, 4, 6])
每个查询的权重和: tensor([[1., 1., 1., 1.],
                          [1., 1., 1., 1.]])

批次 0、查询 0 的注意力权重:
tensor([0.1842, 0.0823, 0.2451, 0.1253, 0.1891, 0.1740])
```

---

## 5. 演进：从 Bahdanau 到 Transformer

![注意力演进时间线](./images/day03/attention-evolution-timeline.png)
*图 6：深度学习中注意力的时间线。从附加机制（2014）到核心架构（2017）。*

### 5.1 Bahdanau 注意力（2014）

原始版本。使用**加性注意力（Additive Attention）**：

```
score(s, h) = v^T · tanh(W_s · s + W_h · h)
```

其中：
- s = 解码器状态
- h = 编码器隐藏状态
- W_s, W_h, v = 学习到的参数

这比点积更具表达力（两个独立的投影），但更慢。

### 5.2 Luong 注意力（2015）

简化为**乘性注意力（Multiplicative Attention）**：

```
score(s, h) = s^T · W · h   （通用）
score(s, h) = s^T · h       （点积）
```

比加性更快，性能相当。"点积"变体变得流行。

### 5.3 Google 神经机器翻译（2016）

Google 的生产系统。使用 8 层 LSTM 配合注意力。达到当时最先进的翻译质量。但仍然是根本性顺序的——无法很好地并行化。

### 5.4 "Attention Is All You Need"（2017）

大飞跃：**完全移除循环**。Transformer 只使用注意力：

1. **自注意力（Self-attention）**：每个位置关注同一序列中的所有其他位置
2. **多头注意力（Multi-head attention）**：多个注意力"头"关注不同方面
3. **位置编码（Positional encoding）**：由于没有循环，位置信息被显式添加

这是我们将在 Day 4 深入探讨的内容。但关键洞察早在 2014 年就在这里了：注意力让我们绕过顺序瓶颈。

---

## 6. 数学推导 [选读]

> 本节面向希望深入理解的读者。可以跳过。

### 6.1 注意力作为加权平均

注意力输出是值的凸组合：

$$
\begin{aligned}
\text{output} &= \sum_{i=1}^{n} \alpha_i v_i \\
\text{其中 } \alpha_i &\geq 0 \text{ 且 } \sum_i \alpha_i = 1
\end{aligned}
$$

这意味着输出总是位于值向量的**凸包**内。注意力不创造新方向——它在现有方向之间插值。

### 6.2 注意力的梯度

让我们计算注意力输出对查询的梯度。这显示了注意力如何学习。

对于单个查询 q 和键 K = [k₁, ..., kₙ]：

$$
\begin{aligned}
s_i &= q^T k_i / \sqrt{d_k} \quad &\text{（分数）} \\
\alpha_i &= \text{softmax}(s)_i \quad &\text{（权重）} \\
o &= \sum_i \alpha_i v_i \quad &\text{（输出）}
\end{aligned}
$$

对 q 的梯度：

$$
\begin{aligned}
\frac{\partial o}{\partial q} &= \sum_i v_i \frac{\partial \alpha_i}{\partial q} \\
\frac{\partial \alpha_i}{\partial q} &= \frac{\partial \alpha_i}{\partial s_j} \frac{\partial s_j}{\partial q} \\
&= \alpha_i (\delta_{ij} - \alpha_j) \cdot \frac{k_j}{\sqrt{d_k}}
\end{aligned}
$$

(δᵢⱼ - αⱼ) 这一项来自 softmax 梯度。这表明：
- 如果 αⱼ 接近 1（高注意力），梯度很小（已经收敛）
- 如果 αⱼ 接近 0，梯度也很小（没有贡献）
- 中间注意力权重时梯度最大

### 6.3 与核方法的联系

注意力和核机器之间有一个美妙的联系。注意力权重可以写成：

$$
\alpha_i = \frac{\exp(q^T k_i / \sqrt{d_k})}{\sum_j \exp(q^T k_j / \sqrt{d_k})} = \frac{K(q, k_i)}{\sum_j K(q, k_j)}
$$

其中 K(q, k) = exp(q^T k / √d_k) 是一个**核函数**（具体来说，是线性核的缩放指数）。

这个视角引出了"线性注意力"和无限上下文 transformer 的研究，我们将在课程后面涉及。

---

## 7. 常见误解

### ❌ "注意力等同于记忆"

注意力和记忆相关但不同。

**注意力**：计算相关性权重的机制。它是单次计算：给定查询，计算键上的权重，返回加权值。

**记忆**：通常意味着跨时间步甚至跨前向传播持久的存储。例如，记忆网络有显式的读/写操作。

在 Transformer 中，单次前向传播内没有持久记忆——一切都从输入新鲜计算。KV 缓存（我们将在 Day 17 讨论）是一种记忆形式，但它是实现优化，不是核心注意力机制的一部分。

### ❌ "注意力使模型可解释"

注意力热图很漂亮且有启发性，但解释是棘手的。

**问题**：
1. 注意力权重并不总是与重要性相关（梯度分析显示）
2. 多个注意力头可能因不同原因关注相同位置
3. 在深层网络中，后面的层建立在变换后的表示上——注意力模式变得更难解释

**适合**：获得关于对齐的粗略直觉。**不可靠于**：细粒度的模型解释。

### ❌ "更高的注意力权重意味着该位置更重要"

不一定。Jain & Wallace (2019) 的研究表明：
- 随机排列注意力权重有时不会损害性能
- 存在对抗性注意力权重（最大不同但相同输出）
- 注意力 ≠ 解释

将注意力可视化作为粗略指南，而不是真相。

---

## 8. 延伸阅读

### 入门
1. **Jay Alammar: Visualizing Attention**（博客）  
   优秀的注意力机制可视化讲解  
   https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

2. **Lilian Weng: Attention? Attention!**（博客）  
   注意力变体的全面概述  
   https://lilianweng.github.io/posts/2018-06-24-attention/

### 进阶
3. **The Illustrated Transformer**（Jay Alammar）  
   Transformer 的权威可视化指南  
   https://jalammar.github.io/illustrated-transformer/

4. **Formal Algorithms for Transformers**（DeepMind）  
   Transformer 操作的数学处理  
   https://arxiv.org/abs/2207.09238

### 论文
5. **Neural Machine Translation by Jointly Learning to Align and Translate**（Bahdanau et al., 2014）  
   将注意力引入 seq2seq 的论文  
   https://arxiv.org/abs/1409.0473

6. **Attention Is All You Need**（Vaswani et al., 2017）  
   Transformer 论文——必读  
   https://arxiv.org/abs/1706.03762

---

## 思考问题

1. **注意力计算值的加权平均。这意味着输出必须位于值向量的"凸包"内。这有什么含义？注意力能"创造"值中没有的信息吗？**

2. **原始的 Bahdanau 注意力使用加性评分，而 Transformer 使用点积。有什么权衡？什么时候加性注意力可能更好？**

3. **我们讨论了注意力权重不能可靠地用于解释。如果你需要理解模型使用了输入的哪些部分，你可能会尝试什么替代方法？**

---

## 总结

| 概念 | 一句话解释 |
|------|-----------|
| 瓶颈问题 | 固定大小的上下文向量无法编码长序列 |
| 注意力机制 | 基于查询-键相似度的值加权和 |
| 查询、键、值 | 我想要什么、有什么可用、检索什么 |
| Softmax 归一化 | 将分数转换为概率分布 |
| 缩放点积 | 除以 √d_k 防止梯度问题 |
| 注意力权重 | 可（粗略）解释为对齐/相关性 |

**核心要点**：注意力通过让解码器直接访问所有编码器状态来解决瓶颈问题。不是把所有内容压缩到一个向量中，而是计算动态的、基于内容的权重，指示哪些源位置与每个输出步骤相关。缩放点积公式成为标准，因为它快速（可并行化的矩阵运算）且数值稳定（√d_k 缩放）。

明天，我们将看到 Transformer 如何将注意力推向更远——**自注意力**让每个位置关注所有其他位置，**多头注意力**捕获不同关系，以及使 GPT 和 BERT 成为可能的完整架构。

---

*Day 3 of 60 | LLM 基础课程*  
*字数: ~3500 | 阅读时间: ~17 分钟*
