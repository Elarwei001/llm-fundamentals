# Day 27: 知识蒸馏——把大模型的智慧压缩进小模型

> **核心问题**：小模型如何学到"大模型的推理过程"，而不只是照抄答案？

---

## 开篇

想象你是一位有30年经验的老医生，带了一个实习医生跟诊一年。如果实习生只抄你的处方，他学到的是"你开了什么药"——而不是"为什么这么开"。真正的知识藏在你的思考过程里：哪些症状你更重视，哪些诊断你会优先排除，你注意到了哪些微妙的模式。

**知识蒸馏（Knowledge Distillation, KD）** 解决的正是机器学习中的这个问题。这个概念最早由 Geoffrey Hinton 和同事在 2015 年的论文 ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) 中提出，核心思想看似简单：不只是让小模型（"学生"）在原始数据上训练，而是让它从更大的、能力更强的模型（"老师"）身上学习——捕捉的不只是最终答案，而是编码在老师概率分布中的**推理过程**。

2025年，这项技术从学术方法变成了头条新闻。DeepSeek 用蒸馏训练了 R1 系列推理模型，从 671B 参数的老师模型中蒸馏出 1.5B 到 70B 的小模型，保留了大部分推理能力。Meta 把 Llama 蒸馏到 1B 和 3B 用于手机端部署。而从专有模型蒸馏是否构成"窃取"的争议，甚至成了地缘政治问题。

让我们来看看它怎么工作的、为什么重要，以及最新的发展。

---

## 1. 什么是知识蒸馏？

### 1.1 核心问题

大模型很贵。GPT-4、Claude、DeepSeek-R1 的训练成本高达数百万美元，推理也需要庞大的 GPU 集群。但很多应用场景——手机端、边缘设备、成本敏感的生产环境——需要小而快的模型，同时还得足够聪明。

知识蒸馏就是一种方案：把大老师模型的"知识"转移到小学生模型里。

#### 直觉：师父带徒弟

把老师模型想象成一位老师傅，学生模型是学徒。学徒不只是复制最终产品——他们观察师傅的工作过程，学习技术，理解决策逻辑。类似地，在知识蒸馏中，学生不只学习最终答案，而是学习老师对**所有可能答案**的置信度分布。

![知识蒸馏架构](./images/day27/kd-architecture-overview.png)
*图1：知识蒸馏架构，展示从老师到学生的三种知识传递方式。*

为了帮助你快速区分这三种方式，可以先看下面这张对比表：

| 蒸馏方式 | 学的是什么 | 需要拿到老师的什么信息 | 优点 | 局限 | 代表脉络 |
|---|---|---|---|---|---|
| **Response-based** | 老师最终生成的答案 / 解释 / 推理轨迹 | 最终输出文本或标签 | 最容易落地，特别适合 API teacher 或闭源大模型 | 信息密度较低，学不到完整内部概率结构 | instruction distillation、synthetic data supervision |
| **Logit-based** | 老师对所有候选输出的概率分布 | logits / soft targets | 能传递暗知识，是最经典的现代 KD 形式 | 闭源模型往往拿不到 logits | Hinton-style distillation |
| **Feature-based** | 老师中间层表示、hidden states、attention 等 | 中间层特征 | 可以让学生学习老师内部表示方式 | 对齐复杂，teacher/student 结构差异大时更难做 | FitNets、representation matching |

图里提到的 temperature softmax，可以先写成这个公式：

$$
P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中：
- $z_i$ 是第 $i$ 个类别的 logit；
- $T$ 是温度；
- $P_i$ 是 softmax 后的概率。

直觉上：
- **$T=1$** 时，就是普通 softmax；
- **$T>1$** 时，分布会更平滑，老师对“次优答案”的看法会更明显；
- **$T<1$** 时，分布会更尖锐，更接近只强调最高概率类别。

### 1.2 为什么不直接训练小模型？

你可能会问：如果需要小模型，为什么不直接在小模型上训练？三个原因：

1. **数据效率**：老师已经学到了丰富的表征。学生从这些表征中学习，比从原始数据中学习高效得多。
2. **暗知识（Dark Knowledge）**：老师对**错误答案**的概率分布也包含有用信息。如果老师认为答案 B 有 70% 概率、答案 C 有 20%、答案 D 有 10%，这告诉学生"C 比 D 更接近正确答案"。
3. **正则化**：用平滑的老师分布训练，本身就是一种正则化，帮助学生比硬标签训练泛化得更好。

### 1.3 从模型压缩到大模型蒸馏，这条技术线是怎么演化来的？

如果直接看到今天的 response-based、logit-based、feature-based 蒸馏，读者很容易觉得这些分类是突然出现的。其实不是。**知识蒸馏本身经历了一条很清楚的技术演化线。**

#### 直觉：先是“压缩模型”，后来才变成“传递知识”

最早的出发点并不是“让学生学老师的推理过程”，而是一个更工程的问题：

> 大模型太大、太慢、太贵，能不能把它的能力压缩到一个更小、更便宜的模型里？

后来研究者逐渐发现，小模型不只是可以学老师的**最终答案**，还可以学：
- 老师的**软概率分布**，
- 老师的**中间表示**，
- 甚至在大模型时代，直接学老师生成出来的**回答与推理轨迹**。

#### 四个关键阶段

这里的“阶段”更准确说是**四条关键演化线索**，不是严格按年份完全线性展开的流水线。尤其是 **FitNets（2014）** 虽然早于 **Hinton 的 2015 distillation 论文**，但它代表的是“学习中间层表示”这条路线的成熟化，而 Hinton 那篇则奠定了后来最经典、最标准的 **logit-based KD** 表述。

| 线索 | 核心思想 | 代表提出者 / 机构 | 年份 | 代表论文 |
|---|---|---|---|---|
| **线索 1：Model Compression** | 用大模型生成更丰富的监督信号，再训练小模型逼近它 | **Cristian Buciluǎ、Rich Caruana、Alexandru Niculescu-Mizil**，IBM Research / Cornell 等 | 2006 | [Model Compression](https://dl.acm.org/doi/10.1145/1150402.1150464) |
| **线索 2：现代 logit-based KD 成型** | 用 temperature-softened logits 把“暗知识”传给学生 | **Geoffrey Hinton、Oriol Vinyals、Jeff Dean**，Google | 2015 | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) |
| **线索 3：学习中间层表示** | 不只学输出，还学 hidden states / intermediate features | **Adriana Romero 等**，Université de Montréal / MILA | 2014 | [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) |
| **线索 4：LLM 时代的 response distillation** | 在拿不到 logits / hidden states 时，直接用 teacher outputs 做监督 | OpenAI、Google、Anthropic、Meta、DeepSeek 等大模型实践路线 | 2022-2025 | 不是单一奠基论文，更像 instruction distillation / synthetic data / teacher-output supervision 的工程汇流 |

#### 线索 1：蒸馏的前身，其实是模型压缩

在 **Buciluǎ、Caruana、Niculescu-Mizil** 2006 年的 **Model Compression** 里，核心问题是：

> 能不能让一个小模型去模仿一个大型集成模型或复杂模型的行为？

这时“蒸馏”这个术语还没有成为今天的标准表达，但思想已经非常接近了：**让大模型产生更细腻的监督信号，让小模型去逼近它。**

#### 线索 2：Hinton 把现代知识蒸馏真正讲清楚了

到了 **Hinton、Vinyals、Dean（Google, 2015）**，知识蒸馏才真正形成今天大家熟悉的标准表述。最核心的贡献有两个：

1. 明确提出用 **soft targets** 来训练 student；
2. 用 **temperature softmax** 揭示老师分布中的 **dark knowledge**。

这里两个概念值得展开一下：

- **soft targets**：传统监督学习里的标签通常是 hard targets，比如“猫”就是 one-hot 的 `[1, 0, 0, 0]`。而 soft targets 则不是只告诉 student “正确答案是谁”，还告诉它 teacher 认为其他候选答案分别有多接近正确答案。比如 teacher 可能认为猫是 0.62、狗是 0.25、狐狸是 0.10、车是 0.03。这样 student 学到的不只是最后答案，还学到类别之间的相似性结构。
- **temperature softmax**：它的公式是

$$
P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $z_i$ 是第 $i$ 个类别的 logit，$T$ 是温度。直觉上，**$T=1$** 是普通 softmax，**$T>1$** 会让分布更平滑，从而把 teacher 对“次优答案”的看法放大出来；而 **$T<1$** 则会让分布更尖锐，更接近只强调最高概率类别。蒸馏里通常关心的是前者，因为它能把隐藏在老师输出里的“暗知识”暴露得更清楚。

这也是为什么今天一说到 KD，大家首先想到的就是 logits、temperature 和 KL divergence。

#### 线索 3：研究者发现，只学输出还不够

之后大家意识到：

> 学生不一定只需要学老师“最后怎么答”，也可以学老师“中间是怎么表示输入的”。

这条线的代表是 **FitNets（Romero et al., 2014）**。它提出用老师中间层作为 **hint** 去指导学生中间层学习。再往后，attention transfer、representation matching 等方法都属于这条思路的延展。

#### 线索 4：大模型时代，response-based distillation 变得更主流

到了 LLM 时代，情况发生了很大变化。

很多时候 teacher 是：
- 超大闭源模型，
- API 形式提供，
- 拿不到 logits，
- 更拿不到 hidden states。

于是最现实的做法就变成：

> 直接让老师生成答案、解释、推理轨迹，再用这些输出去训练学生。

这就是为什么今天在大模型工程实践里，**response-based distillation** 会显得特别常见。它不一定信息密度最高，但通常是**最容易落地**的一种方式。

#### 这条演化线告诉我们什么？

最重要的一点是：

> 今天看到的三种蒸馏方式，并不是平地冒出来的三类标签，而是从“压缩模型”到“传递 soft distribution”，再到“对齐中间表示”，最后到“大模型 teacher 输出监督”逐步长出来的。

所以，如果你把知识蒸馏看成一条演化中的技术路线，而不是一个静态概念，后面的各种变体就会容易理解得多。

---

## 2. 机制：蒸馏是怎么工作的？

### 2.1 关键洞察——"暗知识"

Hinton 的精彩观察是：神经网络 softmax 输出包含的信息远不止最高预测。看一个图像分类的例子：

| 类别 | 原始 Logit | 概率 (T=1) | 概率 (T=5) |
|------|-----------|-------------|-------------|
| 猫 | 5.0 | 0.924 | 0.468 |
| 狗 | 3.0 | 0.125 | 0.254 |
| 汽车 | 1.0 | 0.017 | 0.104 |
| 树 | 0.5 | 0.010 | 0.082 |
| 鸟 | 0.1 | 0.007 | 0.070 |

在温度 T=1 时，分布被"猫"主导——你会以为模型只知道"这是猫"。但在 T=5 时，完整画面浮现：模型认为狗比汽车、树、鸟更可能。非目标类之间的关系就是**"暗知识"**——在硬标签中不可见，但编码在软概率中。

![温度与暗知识](./images/day27/temperature-dark-knowledge.png)
*图2：升高温度 T 如何揭示隐藏在老师输出分布中的"暗知识"。*

### 2.2 蒸馏损失函数

学生用两个损失的组合来训练：

$$
\begin{aligned}
L_{total} &= \alpha \cdot L_{soft} + (1 - \alpha) \cdot L_{hard} \\
L_{soft} &= T^2 \cdot KL\bigl(\sigma(z_t / T) \;\mid\mid\; \sigma(z_s / T)\bigr) \\
L_{hard} &= CE\bigl(\sigma(z_s), y_{true}\bigr)
\end{aligned}
$$

其中：
- $z_t$ = 老师 logits，$z_s$ = 学生 logits
- $T$ = 温度（通常 2-20）
- $\sigma$ = softmax 函数
- $\alpha$ = 软损失和硬损失之间的权重
- KL = Kullback-Leibler 散度
- CE = 交叉熵损失

#### 直觉：为什么要乘以 T²？

当我们把 logits 除以温度 T 时，梯度变成原来的 $1/T$。乘以 $T^2$ 确保软损失的梯度量级正确，防止在 T 较大时被硬损失淹没。

![蒸馏损失函数](./images/day27/kd-loss-function.png)
*图3：知识蒸馏损失函数结合了软蒸馏损失和硬标签损失。*

### 2.3 三种蒸馏层级

并非所有蒸馏都一样。根据你对老师模型的访问权限，有不同方法：

| 方法 | 所需权限 | 传递什么 | 例子 |
|------|---------|---------|------|
| 基于响应 | 黑盒（仅 API） | 老师的最终输出/预测 | Alpaca, Vicuna |
| 基于 Logit | 白盒（logits） | 软化的概率分布 | DistilBERT |
| 基于特征 | 白盒（内部状态） | 隐藏状态、注意力图 | TinyBERT |

![蒸馏方法对比](./images/day27/kd-methods-comparison.png)
*图4：四种蒸馏方法的对比，包括访问要求和实际案例。*

---

## 3. LLM 时代的蒸馏

### 3.1 基于响应的蒸馏：务实路线

最简单的蒸馏形式：用老师模型（通过 API）生成回复，然后在那些回复上微调学生。这是**黑盒蒸馏**——不需要访问老师内部状态。

**著名案例：**
- **Alpaca**（斯坦福，2023年3月）：用 GPT-3.5 生成的 5.2 万条指令微调 Llama 7B。训练成本约 600 美元，达到接近 GPT-3.5 的质量。
- **Vicuna**（LMSYS，2023年3月）：在 7 万条 ShareGPT 的 ChatGPT 对话上训练。
- **OpenHermes**：从 GPT-4 的多任务输出中蒸馏。

局限？学生只学到了老师的**最终答案**，而不是推理过程。就像临摹大师的画作，却没有看过大师作画。

### 3.2 基于 Logit 的蒸馏：捕捉推理

当你能访问老师的 logits（softmax 之前的原始分数），就可以训练学生匹配完整的概率分布。这传递了丰富得多的信息。

**DistilBERT**（Hugging Face，2019年）是一个里程碑：从 BERT-base（110M 参数）蒸馏出 66M 参数的学生，保留了 97% 的性能，同时快 40%、小 60%。

### 3.3 On-Policy 蒸馏：新前沿

2025-2026 年的重大进化是**在线策略蒸馏（On-Policy Distillation）**：学生自己生成回复，老师对学生的回复提供反馈——而不是学生简单模仿老师生成的输出。

#### 直觉：在做中学 vs. 在看中学

传统（离线策略）蒸馏就像看大厨做菜，然后试图复制。在线策略蒸馏就像你自己下厨，然后让大厨品尝并点评**你做的菜**。反馈更有针对性，因为它针对的是**你特有的错误**。

DeepSeek-R1 用了这个方法：蒸馏模型（基于 Qwen2.5 和 Llama3）的训练方式是让学生生成推理链，然后用老师的反馈来改进。结果令人印象深刻：

![DeepSeek-R1 蒸馏性能](./images/day27/deepseek-r1-distill-performance.png)
*图5：DeepSeek-R1 蒸馏模型表明，即使是小模型（7B-70B）也能保留大量推理能力。*

### 3.4 DeepSeek-R1 蒸馏家族（2025年1月）

DeepSeek 开源了一系列蒸馏模型，从 1.5B 到 70B 参数，全部从 671B 参数的 DeepSeek-R1 老师蒸馏而来。关键结果：

| 模型 | 基座 | MATH-500 | GPQA Diamond | 关键发现 |
|------|------|----------|--------------|---------|
| R1（老师） | DeepSeek-V3 MoE | 97.3% | 71.5% | 完整推理能力 |
| Distill-Llama-70B | Llama-3.3-70B | 94.5% | 65.2% | 保留了老师 97% 的数学能力 |
| Distill-Qwen-32B | Qwen2.5-32B | 92.0% | 62.1% | 中端强力选项 |
| Distill-Qwen-14B | Qwen2.5-14B | 86.7% | 56.7% | 可在消费级 GPU 上运行 |
| Distill-Qwen-7B | Qwen2.5-7B | 78.3% | 49.1% | 可部署在边缘设备 |
| Distill-Qwen-1.5B | Qwen2.5-1.5B | 55.6% | 28.9% | 可部署在手机端 |

70B 蒸馏模型在 MATH-500 上得了 94.5%——接近 671B 老师的 97.3%。参数量减少了 10 倍，数学性能只损失 3%。

---

## 4. 争议：蒸馏是"窃取"吗？

2025 年初，一场重大争议爆发。OpenAI 和 Anthropic 指控多家中国 AI 公司——包括 DeepSeek——使用蒸馏"窃取"GPT-4 和 Claude 的能力。指控内容：这些公司通过大量 API 查询生成训练数据，把专有模型的能力蒸馏到自己的开源模型中。

**技术现实是复杂的：**

- **从 API 做黑盒蒸馏技术上并不难。** 任何有 API 访问权限的人都能生成大量老师输出。
- **服务条款通常禁止这种做法。** OpenAI 和 Anthropic 明确禁止用其输出训练竞争模型。
- **很难检测。** 除非蒸馏模型复现了显著特征模式，否则证明蒸馏很困难。
- **DeepSeek 的案例更复杂。** V3/R1 主要通过创新技术（MoE、MLA、GRPO）从头训练，但有证据表明 GPT-4 的输出可能被混入了训练数据。

这场争论在 2026 年仍在继续。DeepSeek V4（2026年4月发布）使用多阶段在线策略蒸馏，从自己的领域专家模型蒸馏，减少了对外部老师的依赖。

---

## 5. 代码示例：简单的基于 Logit 的蒸馏

这是一个最小化的 PyTorch 实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, 
                      temperature=5.0, alpha=0.7):
    """
    计算知识蒸馏损失。
    
    参数:
        student_logits: 学生模型的原始输出 (batch_size, num_classes)
        teacher_logits: 老师模型的原始输出 (batch_size, num_classes)
        labels: 真实标签 (batch_size,)
        temperature: Softmax 温度（越高分布越平滑）
        alpha: 软损失与硬损失的权重（0=只用硬标签，1=只用软标签）
    """
    # 软损失：软分布之间的 KL 散度
    # 乘以 T^2 保持梯度量级
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') 
    soft_loss = soft_loss * (temperature ** 2)
    
    # 硬损失：标准交叉熵
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 组合
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return total_loss


# 使用示例
batch_size, num_classes = 32, 1000
student_logits = torch.randn(batch_size, num_classes)
teacher_logits = torch.randn(batch_size, num_classes)
labels = torch.randint(0, num_classes, (batch_size,))

loss = distillation_loss(student_logits, teacher_logits, labels, 
                         temperature=5.0, alpha=0.7)
print(f"蒸馏损失: {loss.item():.4f}")
```

---

## 6. 数学推导：为什么软目标有效

> 本节适合想深入理解数学原理的读者。

标准交叉熵损失用硬标签只提供 $-\log(p_{true})$——只有正确类的信息。但老师的软目标分布 $q$ 提供了**所有**类的信息。

考虑 KL 散度对某个学生 logit $z_s^{(k)}$ 的梯度：

$$
\begin{aligned}
\frac{\partial}{\partial z_s^{(k)}} KL\bigl(\sigma(z_t / T) \;\mid\mid\; \sigma(z_s / T)\bigr) 
&= \frac{1}{T}\bigl(\sigma(z_s / T)^{(k)} - \sigma(z_t / T)^{(k)}\bigr)
\end{aligned}
$$

这意味着**每个类对梯度的贡献与学生和老师概率之差成正比**。硬标签只有正确类贡献。软目标下，即使不太可能的类（老师分配了 1-5% 概率）也会把学生推向有信息量的方向。

在高温极限（$T \to \infty$）下，蒸馏损失等价于最小化老师和学生 logits 之间的均方误差：

$$
\begin{aligned}
L_{distill} \approx \frac{1}{2} \sum_k (z_t^{(k)} - z_s^{(k)})^2
\end{aligned}
$$

这就是为什么温度是关键超参数：低 T 聚焦于最可能的类，高 T 将注意力分散到所有类。

---

## 7. 常见误解

### ❌ "蒸馏就是抄老师的输出"

不是。基于 logit 和基于特征的蒸馏传递的是老师的**不确定性结构**——它如何思考所有可能的答案，而不仅仅是选了哪个。这比复制输出丰富得多。

### ❌ "蒸馏后的模型一定比老师差"

虽然学生通常在完全相同的分布上无法超越老师，但蒸馏学生常常以**不同方式**泛化。在某些情况下，学生在特定窄任务上甚至超过老师，因为蒸馏过程起到了正则化的作用。

### ❌ "蒸馏需要老师的权重"

只有基于 logit 和基于特征的蒸馏需要。基于响应（黑盒）的蒸馏只需要 API 访问老师的输出——这也是为什么它争议这么大。

### ❌ "蒸馏只是为了模型压缩"

蒸馏还用于：
- **任务迁移**：从通用模型蒸馏出任务专用模型
- **跨模态迁移**：从文本模型向视觉模型传递知识
- **集成压缩**：把多个模型的集成压缩成一个模型
- **推理迁移**：教小模型链式推理（如 DeepSeek-R1）

---

## 8. 延伸阅读

### 入门
1. ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) — Hinton 等人，2015年。开山之作。
2. ["DistilBERT"](https://arxiv.org/abs/1910.01108) — Sanh 等人，2019年。BERT 蒸馏的实践案例。

### 进阶
1. ["TinyBERT"](https://arxiv.org/abs/1909.10351) — Jiao 等人，2019年。两阶段蒸馏：预训练 + 任务特定。
2. ["A Survey of On-Policy Distillation for LLMs"](https://arxiv.org/abs/2604.00626) — Song 等人，2026年4月。在线策略蒸馏方法综述。
3. ["Knowledge Distillation and Dataset Distillation of LLMs"](https://arxiv.org/abs/2504.14772) — 2025年4月综述，覆盖 KD 和数据集蒸馏。

### 论文
1. ["DeepSeek-R1 Technical Report"](https://arxiv.org/abs/2501.12948) — 2025年1月。推理蒸馏的详细技术报告。
2. ["Re-Distilling Smaller DeepSeek R1 Models"](https://dropbox.github.io/r1_redistill_blogpost/) — Dropbox，2025年。第二阶段 logit 对齐显著改进蒸馏模型。
3. ["Knowledge Distillation for LLMs"](https://arxiv.org/abs/2603.13765) — La Torre，2026年3月。Qwen 蒸馏实验。

---

## 思考题

1. 如果一个学生模型能以 1/10 的大小匹配老师 97% 的性能，这说明"知识"到底存在哪里——参数里，还是训练数据里？

2. 在线策略蒸馏（学生生成、老师纠正）似乎比离线策略（老师生成、学生模仿）更有效。为什么从自己的错误中学习可能比复制别人的成功更高效？

3. 如果从专有模型蒸馏被法律限制，开源 AI 生态会怎样？这会不会在"原始训练者"和其他人之间造成永久鸿沟？

---

## 总结

| 概念 | 一句话解释 |
|------|-----------|
| 知识蒸馏 | 从大老师模型向小学生模型传递知识 |
| 暗知识 | 老师对错误答案的概率分布中隐藏的信息 |
| 温度 (T) | 控制 softmax 的平滑程度；T 越高暴露越多暗知识 |
| 基于响应的蒸馏 | 黑盒方法：在老师输出上训练学生（仅需 API） |
| 基于 Logit 的蒸馏 | 白盒方法：匹配老师软化的概率分布 |
| 基于特征的蒸馏 | 白盒方法：匹配中间表征（隐藏状态） |
| 在线策略蒸馏 | 学生生成，老师纠正——更有针对性的学习信号 |
| DeepSeek-R1 Distill | 1.5B-70B 的模型家族，保留高达 97% 的 671B 老师推理能力 |

**核心收获**：知识蒸馏不仅是模型压缩——它是跨尺度传递推理能力的基础技术。隐藏在老师对**错误**答案的概率分布中的"暗知识"，提供了比硬标签丰富得多的训练信号。在 2025-2026 年，蒸馏已成为 AI 生态的核心——从创建可部署的小模型，到谁有权向谁学习的地缘政治。

---

*Day 27 of 60 | LLM Fundamentals*
*Word count: ~2200 | Reading time: ~12 minutes*
