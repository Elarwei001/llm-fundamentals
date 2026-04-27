# Day 25: 评估与基准测试

> **核心问题**: 我们怎么知道一个 LLM 到底好不好——以及为什么最受欢迎的基准测试正在变得不可靠？

---

## 开篇

想象你在买手机。两款机型：一个"手机跑分"98 分，另一个 92 分。选前者，对吧？但如果"手机跑分"只测自家 App 的启动速度呢？98 分那款可能只是在刷分。

这基本就是 2026 年 LLM 评估面临的情况。最著名的基准测试 MMLU（Massive Multitask Language Understanding）已经饱和——顶级模型全在 90% 以上。HumanEval 这个编程基准，模型们跑到了 95%+。当人人都能考高分，考试就失去了区分度。

这篇文章讲的是：我们怎么衡量 LLM、为什么衡量比看起来难得多、以及当旧基准失效新基准崛起时，评估格局是什么样的。

![图 1: LLM 评估基准的分类全景图](./images/day25/benchmark-landscape.png)
*图 1: LLM 基准测试的主要类别——知识与推理、编程、人类偏好——各有不同的评估目标。*

---

## 1. 为什么评估重要（以及为什么困难）

#### 大白话：成绩单问题

把 LLM 评估想象成给学生打分。选择题能测出学生记没记住知识点，但测不出思考能力。论文考试能看出思考水平，但评分有主观性。小组项目能展示协作，但可能一个人扛了整个团队。

LLM 面对同样的问题：没有一个测试能涵盖所有重要维度。

### 1.1 我们想衡量什么

大致来说，LLM 评估沿着几个轴展开：

| 评估轴 | 衡量什么 | 代表性基准 |
|--------|---------|-----------|
| 知识 | 事实和世界知识 | MMLU, TriviaQA |
| 推理 | 逻辑和数学思维 | GSM8K, AIME, ARC-AGI |
| 编程 | 写代码和理解代码 | HumanEval, SWE-bench |
| 指令遵循 | 是否按用户要求去做 | IFEval |
| 安全性 | 避免有害输出 | TruthfulQA, ToxiGen |
| 人类偏好 | 人们实际更喜欢哪个 | Chatbot Arena |

问题是？这些轴并不独立。编程好的模型可能在指令遵循上表现一般。安全的模型可能拒绝太多合理的请求。没有一个数字能告诉你"这个模型更好"。

### 1.2 评估流水线

实际评估是怎么做的？

![图 2: 从基准选择到评分的评估流水线](./images/day25/evaluation-pipeline.png)
*图 2: 标准评估流水线——从选择基准、运行推理到评分比较。*

基本流程是：

1. **选择基准**——选能反映你关心的能力的任务
2. **准备提示**——把问题格式化成模型期望的输入格式
3. **运行推理**——生成模型输出（通常 temperature=0 以确保确定性）
4. **收集输出**——汇总所有响应
5. **评分比较**——计算指标、排名模型

听起来简单，但每一步都藏着复杂性——尤其是评分。

---

## 2. 主要基准测试

### 2.1 MMLU——知识标准（已饱和）

MMLU（Massive Multitask Language Understanding），2021 年由 Hendrycks 等人提出，测试 57 个学科的知识——从抽象代数到兽医学。四选一的多选题。

**为什么重要**：MMLU 是第一个全面的多领域知识测试。多年来，它是每个人引用的数字。

**为什么正在过时**：到 2025-2026 年，前沿模型都跑到了 88-94%。当第一名和第五名差 3 个百分点时，MMLU 就不再有区分度了。

应对方案是 **MMLU-Pro**（更难的题，10 个选项而非 4 个）和 **MMLU-CF**（无污染版本，重新措辞的问题）。MMLU-Pro 相比原版 MMLU 准确率下降 16-33%——这告诉你原版分数有多少水分。

$$
\begin{aligned}
\\text{MMLU Accuracy} &= \\frac{\\text{正确回答}}{\\text{总题数}} \\\\
\\text{MMLU-Pro Accuracy} &\\approx \\text{MMLU Accuracy} - 0.20 \\text{ 到 } 0.30
\\end{aligned}
$$

### 2.2 GPQA——研究生级别科学题

GPQA（Google-Proof Question Answering），由 Rein 等人 2023 年提出，题目难度高到拥有博士学位的人用 Google 都答不好。"Diamond"子集是最难的。

**为什么重要**：这是少数仍然能清晰区分模型的基准之一。截至 2026 年 4 月，最高分大约 65-75%——远未饱和。想知道哪个模型在科学推理上最强，看 GPQA Diamond。

### 2.3 HumanEval——编程基准（接近饱和）

HumanEval，来自 Chen 等人（2021），给模型函数签名和文档字符串，然后检查生成的代码是否通过单元测试。核心指标是 **pass@k**——k 个生成方案中至少一个正确的概率。

$$
\\text{pass@k} = 1 - \\frac{\\binom{n-c}{k}}{\\binom{n}{k}}
$$

其中 n 是总样本数，c 是正确样本数。

**为什么重要**：简单、客观、可自动化。代码能跑就是能跑。

**为什么正在过时**：任务是独立的小函数。现实编程要理解大型代码库、调试、多文件改动。这就是 **SWE-bench** 的价值——它要求模型解决真实 GitHub 项目中的 issue。2026 年，SWE-bench Verified 是前沿模型编程能力的首选基准。

### 2.4 ARC-AGI——抽象推理

ARC-AGI（Abstraction and Reasoning Corpus），由 François Chollet 创建，通过视觉模式谜题测试流体智力。模型必须从少量示例中推断变换规则，然后应用到新输入上。

2025 年发布的 ARC-AGI 2 提高了难度。它专门测试泛化能力而非记忆——谜题是程序化生成的，无法记忆。

### 2.5 Humanity's Last Exam (HLE)

2025 年初发布，由 Center for AI Safety 和 Scale AI 合作完成。包含超过 12,000 道专家设计的题目，覆盖 14 个领域。名字很夸张，但基准是真的——设计目标就是让前沿模型也感到困难。2026 年，顶级模型得分远低于 50%，是最佳区分性基准之一。

### 2.6 Chatbot Arena——大规模人类偏好

Chatbot Arena 由 LMSYS 运营，采用完全不同的方式：真实人类与两个匿名模型并排对话，投票选择更喜欢哪个回复。结果汇总为 **Elo 评分**，和象棋用的是同一套系统。

**Elo 怎么算**：每次模型之间的"对战"就是一次用户投票。低评分模型赢了高评分模型，涨分更多——和象棋一样。更新规则：

$$
R_{\\text{new}} = R_{\\text{old}} + K \\cdot (S - E)
$$

其中 R 是评分，K 是敏感度常数（通常为 32），S 是实际结果（赢=1，输=0，平=0.5），E 是基于当前评分的期望得分：

$$
E = \\frac{1}{1 + 10^{(R_{\\text{对手}} - R_{\\text{模型}}) / 400}}
$$

截至 2026 年 4 月，Arena Elo 最高分超过 1400。Claude Opus 4.6 在编程专项排行榜上突破 1560，是首个超过 1500 的模型。

**局限性**：人类偏好有噪声。用户可能偏好更长、更自信的回答，即使内容有误。用户群体可能不代表你的具体使用场景。

### 2.7 为什么没有单一基准够用

不同基准测试不同的东西。一个模型可能统治 MMLU 但在 ARC-AGI 上挣扎。这就是为什么从业者越来越多地使用**基准画像**——雷达图，同时展示多个基准的表现。

![图 3b: 两个模型在七个基准上的雷达图对比](./images/day25/benchmark-radar-comparison.png)
*图 3b: 两个假设模型在七个基准上的对比。模型 A 在 MMLU 和 HumanEval 上领先（已饱和），但 SWE-bench 和 HLE 上的差距揭示了真正的差异。*

关键洞察：**看形状，不看单个数字**。MMLU 高但 SWE-bench 低的模型是知识引擎但不会写代码。Arena Elo 高但 GPQA 低的模型可能讨喜但推理深度不够。

---

## 3. 基准饱和问题

![图 3: 基准随时间饱和的趋势](./images/day25/benchmark-saturation-timeline.png)
*图 3: 基准饱和时间线——MMLU 和 HumanEval 已有效饱和，而 GPQA Diamond 仍能区分模型。*

这是 LLM 评估中最重要的动态之一。模式如下：

1. 新基准发布 → 模型表现很差
2. 模型建造者针对它优化 → 分数快速上升
3. 分数逼近天花板 → 基准不再有区分度
4. 更难的基准被创建 → 循环重复

**MMLU** 从 GPT-2 时代的 ~25%（四选一随机猜就是 25%）到前沿模型的 90%+。**HumanEval** 从早期 Codex 的 ~10% 到 95%+。每次基准都变得不太能用来排名模型。

#### 实践中的意义

当你看到一个模型声称"MMLU 上最强"时，要问：
- *MMLU-Pro 呢？*（可能低得多）
- *污染数据集 vs. 干净数据集呢？*（有些题目可能在训练数据里）
- *Chatbot Arena 怎么说？*（人类偏好更难刷分）

---

## 4. 数据污染——房间里的大象

![图 4: 基准数据如何泄漏到训练集](./images/day25/contamination-problem.png)
*图 4: 当基准问题出现在训练语料中时就会发生数据污染，通过记忆而非真实能力获得虚高分数。*

这是基准有效性面临的最严重威胁。

### 4.1 污染如何发生

LLM 在海量网页抓取数据上训练。MMLU 和 HumanEval 等基准在互联网上公开可得。如果基准问题（或非常相似的内容）出现在训练数据中，模型可以记住答案而不是推理答案。

Gema 等人 2025 年的调查发现，污染检测方法一致地发现了流行基准和常见预训练数据集之间的重叠。"Are We Done with MMLU?" 论文记录了多个商业模型显著的测试集泄漏。

### 4.2 为什么难以检测

污染不总是精确匹配。训练数据可能包含：
- 基准问题的**改写版本**
- 论坛和教程中对基准答案的**讨论**
- 来自相同来源材料的**相似但不完全相同**的问题

现代污染检测用 LLM 本身来检查模型是否"认出"了不该见过的测试题，但这是一场军备竞赛。

### 4.3 缓解策略

| 策略 | 原理 | 代价 |
|------|------|------|
| 保密基准 | 不公开部分基准 | 限制了社区参与 |
| 动态基准 | 定期生成新题 | 维护成本高 |
| 改写版本 | 重述现有问题 | 可能改变难度 |
| 污染检测 | 检查训练数据重叠 | 无法检测改写 |
| 实时基准 | 使用实时任务（SWE-bench） | 标准化更难 |

### 4.4 关键论文

- ["Are We Done with MMLU?"](https://arxiv.org/abs/2406.04127)——记录了 MMLU 污染问题
- ["A Survey on Data Contamination for Large Language Models"](https://arxiv.org/abs/2502.14425)——2025 年全面综述
- ["When Benchmarks Leak: Inference-Time Decontamination for LLMs"](https://arxiv.org/abs/2601.19334)——2026 年 1 月，提出实时去污染方案

---

## 5. 现代评估：2026 年该用什么

鉴于饱和和污染问题，到底该用什么？这里有一份实用指南：

### 5.1 基准选择指南

| 你的问题 | 最佳基准 | 为什么 |
|---------|---------|-------|
| 哪个模型总体最好？ | Chatbot Arena Elo | 人类偏好，难刷分 |
| 哪个科学推理最强？ | GPQA Diamond | 未饱和，专家级 |
| 哪个写代码最强？ | SWE-bench Verified | 真实 GitHub issue，不是玩具函数 |
| 哪个数学最强？ | AIME 2025 | 奥赛级别，无法记忆 |
| 哪个最聪明？ | ARC-AGI 2 | 测试流体智力 |
| 哪个最听指令？ | IFEval | 直接测试指令遵循 |
| 最难的前沿在哪里？ | HLE | 专家级，分数很低 |

### 5.2 AAII 指数

Artificial Analysis Intelligence Index (AAII) v3 聚合了 10 个高难度评估：MMLU-Pro、HLE、GPQA Diamond、AIME 等，给出一个综合分数。虽然单一数字不能涵盖一切，但 AAII 提供了前沿模型能力的合理概览。

### 5.3 领域特定评估

通用基准不能告诉你模型在*你的*任务上表现如何。生产环境中：

1. **自建评估集**——收集真实用户查询和标准答案
2. **用 LLM 当评委**——让强模型按你的标准评价输出
3. **A/B 测试真实用户**——终极评估
4. **追踪性能漂移**——即使基准在进步，模型在你的特定场景上可能变差

---

## 6. 常见误解

### ❌ "MMLU 分数更高的模型更聪明"

不一定。MMLU 已饱和且可能被污染。2% 的差异是噪声，不是信号。看 GPQA、ARC-AGI 或 Arena Elo。

### ❌ "基准分数能预测实际表现"

基准表现和生产表现之间的差距是公认的。统治排行榜的模型在真实部署中常常表现不佳，因为基准测试的是狭窄、明确的任务，而真实使用是混乱和开放式的。

### ❌ "我们只需要更难的基准"

更难的基准能暂时帮助，但饱和循环会重演。真正的解决方案是在**你自己的使用场景上评估**，而不是依赖任何通用基准。

### ❌ "Chatbot Arena 是完美的因为用了真人"

人类偏好有噪声、偏向冗长和自信、可能不反映你的具体需求。Arena Elo 是目前最好的通用信号，但仍然不完美。

---

## 7. 代码示例：运行一个简单的评估

以下是如何使用 Hugging Face 数据集运行 MMLU 风格评估：

```python
"""
简单的 MMLU 风格评估，使用 Hugging Face 数据集。
通过模型运行多选题并计算准确率。
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载 MMLU 的一小部分（STEM 学科）
dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
dataset = dataset.filter(lambda x: x["subject"] in ["abstract_algebra", "astronomy"])

# 加载模型和分词器
model_name = "gpt2"  # 替换为你的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

def format_mmlu_prompt(question, choices):
    """将多选题格式化为提示词。"""
    labels = ["A", "B", "C", "D"]
    options = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return f"{question}\n{options}\nAnswer:"

def evaluate_one(question, choices, answer_idx):
    """
    通过比较每个选项的对数概率来评估单个问题。
    """
    prompt = format_mmlu_prompt(question, choices)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # 最后一个 token 的 logits

    # 比较 A, B, C, D token 的对数概率
    label_tokens = [tokenizer.encode(l)[0] for l in ["A", "B", "C", "D"]]
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = [log_probs[t].item() for t in label_tokens]

    predicted = scores.index(max(scores))
    return predicted == answer_idx

# 运行评估
correct = 0
total = 0
for example in dataset.select(range(min(50, len(dataset)))):
    if evaluate_one(example["question"], example["choices"], example["answer"]):
        correct += 1
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
```

这展示了基本模式：格式化问题、获取模型分数、比较预测和答案。生产评估系统会加入提示工程、少样本示例、思维链和更复杂的评分——但核心循环是一样的。

---

## 8. 延伸阅读

### 入门
1. [LMSYS Chatbot Arena](https://chat.lmsys.org)——亲自试试，给模型对比投票
2. [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)——社区基准追踪
3. [LLM Benchmarks Compared (LXT, 2026)](https://www.lxt.ai/blog/llm-benchmarks/)——当前格局的优质概览

### 进阶
1. ["A Survey on Data Contamination for Large Language Models"](https://arxiv.org/abs/2502.14425)——数据污染综合综述
2. ["Are We Done with MMLU?"](https://arxiv.org/abs/2406.04127)——MMLU 局限性分析
3. ["When Benchmarks Leak: Inference-Time Decontamination for LLMs"](https://arxiv.org/abs/2601.19334)——2026 年 1 月的污染缓解方案

### 关键论文
1. ["Measuring Massive Multitask Language Understanding" (MMLU)](https://arxiv.org/abs/2009.03300)——Hendrycks 等人, 2021
2. ["Evaluating Large Language Models Trained on Code" (HumanEval)](https://arxiv.org/abs/2107.03374)——Chen 等人, 2021
3. ["Google-Proof Question Answering" (GPQA)](https://arxiv.org/abs/2311.12022)——Rein 等人, 2023
4. ["Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference"](https://arxiv.org/abs/2403.04132)——Zheng 等人, 2024
5. ["SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"](https://arxiv.org/abs/2310.06770)——Jimenez 等人, 2023

---

## 思考题

1. 如果你在搭建一个客服聊天机器人，你会信任哪些基准来选择合适的模型——为什么光看 MMLU 不够？
2. 你觉得为什么基准饱和发生得这么快？是因为基准设计得不好，还是领域发展太快？
3. 你会如何为一个 LLM 产品设计评估方案，避免这篇文章讨论的各种陷阱？

---

## 总结

| 概念 | 一句话解释 |
|------|-----------|
| MMLU | 多学科知识测试，已饱和至 90%+ |
| GPQA Diamond | 研究生级科学题，仍能区分模型 |
| HumanEval | 函数级编程测试，接近饱和 |
| SWE-bench | 真实 GitHub issue 解决，更适合现代编程评估 |
| ARC-AGI 2 | 视觉模式推理，测试泛化而非记忆 |
| HLE | 14 个领域的专家级题目，分数很低 |
| Chatbot Arena | 人类偏好投票配 Elo 评分，600 万+ 投票 |
| 数据污染 | 基准问题泄漏到训练数据，虚高分数 |
| 基准饱和 | 所有顶级模型得分相近时，基准失去区分度 |

**核心要点**: LLM 评估是基准创建者和模型建造者之间的军备竞赛。旧基准饱和，数据污染破坏有效性，没有单一数字能反映真实表现。2026 年的最佳做法是使用多个互补基准（GPQA、SWE-bench、Arena Elo），并且始终在你自己的具体场景上评估。

---

*Day 25 of 60 | LLM 基础课程*
*字数: ~2900 | 阅读时间: ~14 分钟*
