# LLM Fundamentals 术语表

> 中英术语对照，按首字母排序

## A

| English | 中文 | 简要说明 |
|---------|------|----------|
| Activation Function | 激活函数 | 引入非线性的函数，如 ReLU、Sigmoid |
| ALiBi | ALiBi / 线性偏置注意力 | Attention with Linear Biases，通过距离相关的线性偏置帮助模型外推到更长输入长度 |
| Attention | 注意力机制 | 让模型关注输入的不同部分 |
| Autoregressive | 自回归 | 根据前面的 token 预测下一个 token |
| Abstention | 保留判断 / 拒答 | 在证据不足时选择明确表示不确定或拒绝给出结论 |

## B

| English | 中文 | 简要说明 |
|---------|------|----------|
| Backpropagation | 反向传播 | 计算梯度的高效算法 |
| Batch Size | 批大小 | 每次训练使用的样本数 |
| BERT | BERT | Bidirectional Encoder Representations from Transformers |
| Beam Search | 束搜索 | 保留多个高分候选序列并继续扩展的解码算法 |
| BIG-bench | BIG-bench | Beyond the Imitation Game，包含 200+ 任务的大型语言模型评测基准 |
| BPE | 字节对编码 | Byte Pair Encoding，子词分词算法 |
| Brier Score | Brier 分数 | 概率预测的准确性度量，值越低越好 |

## C

| English | 中文 | 简要说明 |
|---------|------|----------|
| CBOW | 连续词袋模型 | Continuous Bag of Words，Word2Vec 的一种变体，从上下文预测中心词 |
| Chinchilla Scaling | Chinchilla 缩放 | DeepMind 2022 年提出的最优训练配方：每参数约 20 token |
| Compute-Optimal | 计算最优 | 在固定计算预算下达到最低损失的模型规模和数据量配置 |
| Cosine Decay | 余弦衰减 | 学习率调度策略，按余弦曲线从峰值降到接近零 |
| Cross-Entropy Loss | 交叉熵损失 | 衡量预测分布与真实分布差异的损失函数，等价于负对数似然 |
| Curriculum Learning | 课程学习 | 按照从易到难的顺序安排训练样本 |
| Chain Rule of Probability | 概率链式法则 | 将句子概率分解为逐词条件概率乘积的规则 |
| Chain-of-Thought (CoT) | 思维链 | 让模型逐步推理的提示技术 |
| Context Window | 上下文窗口 | 模型能处理的最大 token 数 |
| Cosine Similarity | 余弦相似度 | 通过向量夹角的余弦值衡量相似度，常用于比较嵌入 |
| Cross-Entropy Loss | 交叉熵损失 | 分类任务常用的损失函数 |

## D

| English | 中文 | 简要说明 |
|---------|------|----------|
| Decoder | 解码器 | Transformer 的生成部分 |
| Dense Embedding | 稠密嵌入 | 低维连续向量表示，每个维度都有值，与稀疏的 one-hot 相对 |
| Distributional Hypothesis | 分布式假设 | 语言学假说：出现在相似上下文中的词具有相似含义 |
| Distillation | 知识蒸馏 | 用大模型训练小模型 |
| Dropout | Dropout | 随机丢弃神经元的正则化技术 |

## E

| English | 中文 | 简要说明 |
|---------|------|----------|
| Embedding | 嵌入/向量表示 | 将离散 token 映射到连续向量空间 |
| Emergent Abilities | 涌现能力 | 在特定模型规模突然出现的能力，小模型完全不具备 |
| Emergence | 涌现 | 大模型展现出小模型没有的能力 |
| Encoder | 编码器 | Transformer 的理解部分 |
| Exposure Bias | 暴露偏差 | 训练时看真实前缀、推理时看模型自生成前缀所带来的分布差异 |

## F

| English | 中文 | 简要说明 |
|---------|------|----------|
| Fine-tuning | 微调 | 在预训练模型基础上针对特定任务训练 |
| Faithful Reasoning | 忠实推理 | 可见的推理过程与模型真实导致答案的内部因果过程较一致的性质 |
| FlashAttention | FlashAttention | 内存高效的注意力计算方法 |
| Factuality | 事实性 / 事实一致性 | 输出与真实世界事实或可信证据保持一致的程度 |
| Few-shot Prompting | 少样本提示 | 在 prompt 中提供少量示例，让模型临时学会任务模式 |
| Function Calling | 函数调用 | LLM 调用外部工具的能力 |

## G

| English | 中文 | 简要说明 |
|---------|------|----------|
| Gradient Descent | 梯度下降 | 优化神经网络的核心算法 |
| GPT | GPT | Generative Pre-trained Transformer |
| Grounding | 基于证据对齐 / 落地支撑 | 让模型的回答明确建立在输入文档、检索结果或工具输出之上 |
| Grouped-Query Attention (GQA) | 分组查询注意力 | 多个 query 头按组共享 key/value 的注意力变体，降低缓存开销 |

## H

| English | 中文 | 简要说明 |
|---------|------|----------|
| Hallucination | 幻觉 | LLM 生成虚假但看起来合理的内容 |
| Hidden Layer | 隐藏层 | 神经网络的中间层 |

## I

| English | 中文 | 简要说明 |
|---------|------|----------|
| In-context Learning | 上下文学习 | 通过 prompt 中的示例学习，无需更新参数 |
| Inference | 推理 | 使用训练好的模型进行预测 |
| Inference-Time Scaling | 推理时缩放 | 通过增加推理计算（如更长的思维链）提升性能 |
| Instruction Tuning | 指令微调 | 用指令-回答数据训练模型，使其更好地遵循自然语言指令 |

## K

| English | 中文 | 简要说明 |
|---------|------|----------|
| KV Cache | KV 缓存 | 在推理时缓存历史 token 的 key/value 张量，避免重复计算 |

## L

| English | 中文 | 简要说明 |
|---------|------|----------|
| Large Language Model (LLM) | 大语言模型 | 大规模预训练的语言模型 |
| Learning Rate | 学习率 | 梯度下降的步长 |
| LoRA | LoRA | Low-Rank Adaptation，参数高效微调方法 |
| Logits | Logits / 未归一化分数 | Softmax 之前的原始输出分数，用于计算概率分布 |
| Loss Function | 损失函数 | 衡量预测与真实值差距的函数 |

## M

| English | 中文 | 简要说明 |
|---------|------|----------|
| Measurement Artifact | 测量假象 | 由离散指标（如准确率）导致的表面涌现现象 |
| MLP | 多层感知机 | Multi-Layer Perceptron，基础神经网络 |
| Model-Based Reinforcement Learning | 基于模型的强化学习 | 先学习或利用环境模型，再在模型中规划或生成训练信号的强化学习范式 |
| Model-Predictive Control (MPC) | 模型预测控制 | 先用模型评估多条未来动作序列，只执行当前最优方案的前几步，再重新规划 |
| MoE | 混合专家 | Mixture of Experts，稀疏激活架构 |
| Multi-head Attention | 多头注意力 | 并行计算多组注意力 |
| Multi-Query Attention (MQA) | 多查询注意力 | 多个 query 头共享同一组 key/value 的注意力变体，降低解码内存带宽压力 |

## N

| English | 中文 | 简要说明 |
|---------|------|----------|
| Negative Sampling | 负采样 | Word2Vec 训练技巧，通过区分真假词对避免全词表 softmax |
| Neural Network | 神经网络 | 由神经元组成的计算图 |
| Next Token Prediction | 下一个词预测 | 语言模型的核心训练目标 |
| Nucleus Sampling | 核采样 | Top-p 采样，只保留累计概率达到阈值的最小 token 集合 |

## O

| English | 中文 | 简要说明 |
|---------|------|----------|
| One-Hot Encoding | 独热编码 | 稀疏向量表示，只有一个维度为 1，其余为 0 |
| Overfitting | 过拟合 | 模型在训练集上表现好，测试集上差 |

## P

| English | 中文 | 简要说明 |
|---------|------|----------|
| Paged Attention | 分页注意力 | 将 KV 缓存按块存储和映射的推理优化技术，常用于减少内存碎片 |
| Parameter | 参数 | 模型中可学习的权重 |
| Phase Transition | 相变 | 在临界阈值发生的质的变化（物理学概念，用于类比涌现） |
| Pipeline Parallelism | 流水线并行 | 不同 Transformer 层分布在不同 GPU 上，数据像流水线一样流过 |
| Position Encoding | 位置编码 | 让模型感知 token 顺序 |
| Position Interpolation | 位置插值 | 通过重缩放位置索引，把更长序列压缩映射到模型更熟悉的位置范围 |
| Power Law | 幂律 | 变量间的乘幂关系，如 $y \propto x^{\alpha}$，缩放定律的数学基础 |
| Pre-training | 预训练 | 在大规模数据上的初始训练 |
| Prefix Caching | 前缀缓存 | 复用共享 prompt 前缀的 prefill 结果，避免重复构建 KV 缓存 |
| Prompt | 提示词 | 给模型的输入指令 |
| Prompt Engineering | 提示工程 | 通过设计指令、上下文、示例和输出约束来改善模型行为的方法 |

## Q

| English | 中文 | 简要说明 |
|---------|------|----------|
| Quantization | 量化 | 降低模型精度以减少内存和计算 |

## R

| English | 中文 | 简要说明 |
|---------|------|----------|
| RAG | 检索增强生成 | Retrieval-Augmented Generation |
| ReLU | ReLU | Rectified Linear Unit，常用激活函数 |
| RLHF | 人类反馈强化学习 | Reinforcement Learning from Human Feedback |
| RNN | 循环神经网络 | Recurrent Neural Network |
| RoPE | 旋转位置嵌入 | Rotary Position Embedding，通过复数旋转编码相对位置 |

## S

| English | 中文 | 简要说明 |
|---------|------|----------|
| Scaling Laws | 缩放定律 | 模型性能与规模的关系 |
| Self-Consistency | 自洽采样 / 自一致性 | 采样多条推理路径并通过答案一致性选择更可靠结果的方法 |
| Self-Attention | 自注意力 | Token 之间相互计算注意力 |
| Self-supervised Learning | 自监督学习 | 标签来自数据本身的机器学习方法，无需人类标注 |
| Speculative Decoding | 推测解码 | 用小模型先提议多个 token，再由大模型成批验证以加速推理 |
| Semantic Space | 语义空间 | 嵌入向量所在的高维空间，相似含义的词距离更近 |
| SentencePiece | SentencePiece | Google 的语言无关子词分词器 |
| Skip-gram | Skip-gram | Word2Vec 的一种变体，从中心词预测上下文 |
| Sliding-Window Attention | 滑动窗口注意力 | 每个 token 主要关注固定邻域内的 token，以降低长序列计算和内存成本 |
| Softmax | Softmax | 将 logits 转换为概率分布 |
| SAT Solver | 可满足性求解器 | 用于求解布尔可满足性问题的程序，常用于形式约束检查和组合优化 |
| Subword Tokenization | 子词分词 | 介于字符和词之间的分词策略 |
| Symbolic Perturbation | 符号扰动 | 在不改变底层数学结构的前提下，改写变量名、数字顺序或表面叙述，用于测试推理鲁棒性 |
| Synthetic Data | 合成数据 | 由模型生成的用于训练其他模型的数据 |
| Typical Sampling | 典型采样 | 偏好惊讶度接近分布平均值 token 的解码方法 |

## T

| English | 中文 | 简要说明 |
|---------|------|----------|
| Temperature | 温度 | 控制生成随机性的参数 |
| Test-Time Compute | 测试时计算 | 在推理阶段通过采样、搜索、验证等额外计算提升结果质量的做法 |
| Token | Token | 文本的基本单位 |
| Tokenizer | 分词器 | 将文本转换为 token 的工具 |
| Transformer | Transformer | 基于注意力机制的神经网络架构 |
| Verification Pass | 验证前向 / 验证步骤 | 目标模型对草稿 token 块进行一次性检查的前向计算 |
| Verifier | 验证器 | 用来检查候选答案或中间步骤是否正确、是否有证据支撑的组件 |

## U

| English | 中文 | 简要说明 |
|---------|------|----------|
| Universal Approximation | 通用逼近 | 神经网络能逼近任意连续函数 |

## V

| English | 中文 | 简要说明 |
|---------|------|----------|
| Vanishing Gradient | 梯度消失 | 深层网络中梯度趋近于零的问题 |

## V

| English | 中文 | 简要说明 |
|---------|------|----------|
| Vocabulary | 词表 | 分词器可识别的所有 token 集合 |

## W

| English | 中文 | 简要说明 |
|---------|------|----------|
| Weight | 权重 | 神经网络中的可学习参数 |
| Word Analogy | 词类比 | 嵌入空间中的向量算术，如 king - man + woman ≈ queen |
| Word2Vec | Word2Vec | Google 2013 年提出的词嵌入学习算法 |
| WordPiece | WordPiece | Google BERT 使用的子词分词算法 |
| World Model | 世界模型 | 关于环境潜在状态、状态转移和动作后果的预测模型，常用于规划与控制 |

---

*持续更新中...*
