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

## C

| English | 中文 | 简要说明 |
|---------|------|----------|
| Chain Rule of Probability | 概率链式法则 | 将句子概率分解为逐词条件概率乘积的规则 |
| Chain-of-Thought (CoT) | 思维链 | 让模型逐步推理的提示技术 |
| Context Window | 上下文窗口 | 模型能处理的最大 token 数 |
| Cross-Entropy Loss | 交叉熵损失 | 分类任务常用的损失函数 |

## D

| English | 中文 | 简要说明 |
|---------|------|----------|
| Decoder | 解码器 | Transformer 的生成部分 |
| Distillation | 知识蒸馏 | 用大模型训练小模型 |
| Dropout | Dropout | 随机丢弃神经元的正则化技术 |

## E

| English | 中文 | 简要说明 |
|---------|------|----------|
| Embedding | 嵌入/向量表示 | 将离散 token 映射到连续向量空间 |
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
| MLP | 多层感知机 | Multi-Layer Perceptron，基础神经网络 |
| MoE | 混合专家 | Mixture of Experts，稀疏激活架构 |
| Multi-head Attention | 多头注意力 | 并行计算多组注意力 |

## N

| English | 中文 | 简要说明 |
|---------|------|----------|
| Neural Network | 神经网络 | 由神经元组成的计算图 |
| Next Token Prediction | 下一个词预测 | 语言模型的核心训练目标 |

## O

| English | 中文 | 简要说明 |
|---------|------|----------|
| Overfitting | 过拟合 | 模型在训练集上表现好，测试集上差 |

## P

| English | 中文 | 简要说明 |
|---------|------|----------|
| Parameter | 参数 | 模型中可学习的权重 |
| Position Encoding | 位置编码 | 让模型感知 token 顺序 |
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

## S

| English | 中文 | 简要说明 |
|---------|------|----------|
| Scaling Laws | 缩放定律 | 模型性能与规模的关系 |
| Self-Attention | 自注意力 | Token 之间相互计算注意力 |
| Softmax | Softmax | 将 logits 转换为概率分布 |

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

## W

| English | 中文 | 简要说明 |
|---------|------|----------|
| Weight | 权重 | 神经网络中的可学习参数 |

---

*持续更新中...*
