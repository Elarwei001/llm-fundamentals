# LLM Fundamentals 术语表

> 中英术语对照，按首字母排序

## A

| English | 中文 | 简要说明 |
|---------|------|----------|
| Activation Function | 激活函数 | 引入非线性的函数，如 ReLU、Sigmoid |
| Attention | 注意力机制 | 让模型关注输入的不同部分 |
| Autoregressive | 自回归 | 根据前面的 token 预测下一个 token |

## B

| English | 中文 | 简要说明 |
|---------|------|----------|
| Backpropagation | 反向传播 | 计算梯度的高效算法 |
| Batch Size | 批大小 | 每次训练使用的样本数 |
| BERT | BERT | Bidirectional Encoder Representations from Transformers |
| BIG-bench | BIG-bench | Beyond the Imitation Game，包含 200+ 任务的大型语言模型评测基准 |
| BPE | 字节对编码 | Byte Pair Encoding，子词分词算法 |
| Brier Score | Brier 分数 | 概率预测的准确性度量，值越低越好 |

## C

| English | 中文 | 简要说明 |
|---------|------|----------|
| CBOW | 连续词袋模型 | Continuous Bag of Words，Word2Vec 的一种变体，从上下文预测中心词 |
| Chinchilla Scaling | Chinchilla 缩放 | DeepMind 2022 年提出的最优训练配方：每参数约 20 token |
| Compute-Optimal | 计算最优 | 在固定计算预算下达到最低损失的模型规模和数据量配置 |
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

## F

| English | 中文 | 简要说明 |
|---------|------|----------|
| Fine-tuning | 微调 | 在预训练模型基础上针对特定任务训练 |
| FlashAttention | FlashAttention | 内存高效的注意力计算方法 |
| Function Calling | 函数调用 | LLM 调用外部工具的能力 |

## G

| English | 中文 | 简要说明 |
|---------|------|----------|
| Gradient Descent | 梯度下降 | 优化神经网络的核心算法 |
| GPT | GPT | Generative Pre-trained Transformer |

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

## K

| English | 中文 | 简要说明 |
|---------|------|----------|
| KV Cache | KV 缓存 | 加速自回归生成的技术 |

## L

| English | 中文 | 简要说明 |
|---------|------|----------|
| Large Language Model (LLM) | 大语言模型 | 大规模预训练的语言模型 |
| Learning Rate | 学习率 | 梯度下降的步长 |
| LoRA | LoRA | Low-Rank Adaptation，参数高效微调方法 |
| Loss Function | 损失函数 | 衡量预测与真实值差距的函数 |

## M

| English | 中文 | 简要说明 |
|---------|------|----------|
| Measurement Artifact | 测量假象 | 由离散指标（如准确率）导致的表面涌现现象 |
| MLP | 多层感知机 | Multi-Layer Perceptron，基础神经网络 |
| MoE | 混合专家 | Mixture of Experts，稀疏激活架构 |
| Multi-head Attention | 多头注意力 | 并行计算多组注意力 |

## N

| English | 中文 | 简要说明 |
|---------|------|----------|
| Negative Sampling | 负采样 | Word2Vec 训练技巧，通过区分真假词对避免全词表 softmax |
| Neural Network | 神经网络 | 由神经元组成的计算图 |
| Next Token Prediction | 下一个词预测 | 语言模型的核心训练目标 |

## O

| English | 中文 | 简要说明 |
|---------|------|----------|
| One-Hot Encoding | 独热编码 | 稀疏向量表示，只有一个维度为 1，其余为 0 |
| Overfitting | 过拟合 | 模型在训练集上表现好，测试集上差 |

## P

| English | 中文 | 简要说明 |
|---------|------|----------|
| Parameter | 参数 | 模型中可学习的权重 |
| Phase Transition | 相变 | 在临界阈值发生的质的变化（物理学概念，用于类比涌现） |
| Position Encoding | 位置编码 | 让模型感知 token 顺序 |
| Power Law | 幂律 | 变量间的乘幂关系，如 $y \propto x^{\alpha}$，缩放定律的数学基础 |
| Pre-training | 预训练 | 在大规模数据上的初始训练 |
| Prompt | 提示词 | 给模型的输入指令 |

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
| Self-Attention | 自注意力 | Token 之间相互计算注意力 |
| Semantic Space | 语义空间 | 嵌入向量所在的高维空间，相似含义的词距离更近 |
| SentencePiece | SentencePiece | Google 的语言无关子词分词器 |
| Skip-gram | Skip-gram | Word2Vec 的一种变体，从中心词预测上下文 |
| Softmax | Softmax | 将 logits 转换为概率分布 |
| Subword Tokenization | 子词分词 | 介于字符和词之间的分词策略 |

## T

| English | 中文 | 简要说明 |
|---------|------|----------|
| Temperature | 温度 | 控制生成随机性的参数 |
| Token | Token | 文本的基本单位 |
| Tokenizer | 分词器 | 将文本转换为 token 的工具 |
| Transformer | Transformer | 基于注意力机制的神经网络架构 |

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

---

*持续更新中...*
