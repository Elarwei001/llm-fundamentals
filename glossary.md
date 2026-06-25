# LLM Fundamentals 术语表

> 中英术语对照，按首字母排序

## A

| English | 中文 | 简要说明 |
|---------|------|----------|
| Activation Function | 激活函数 | 引入非线性的函数，如 ReLU、Sigmoid |
| Access Control List (ACL) | 访问控制列表 | 描述哪些用户或角色可以访问某个文档、字段或资源的权限规则 |
| ANN (Approximate Nearest Neighbor) | 近似最近邻 | 在高维空间中快速找到近似最近邻的算法，如 HNSW |
| Agentic Memory (AgeMem) | 智能体记忆 | 统一长短期记忆管理的框架，让智能体学习何时存取和遗忘 |
| Agentic RAG | Agentic RAG | 让 LLM 在回答前迭代搜索、检查证据、改写查询并判断证据是否充分的 RAG 形态 |
| ALiBi | ALiBi / 线性偏置注意力 | Attention with Linear Biases，通过距离相关的线性偏置帮助模型外推到更长输入长度 |
| Attention | 注意力机制 | 让模型关注输入的不同部分 |
| Autoregressive | 自回归 | 根据前面的 token 预测下一个 token |
| Abstention | 保留判断 / 拒答 | 在证据不足时选择明确表示不确定或拒绝给出结论 |
| A2A (Agent-to-Agent Protocol) | Agent-to-Agent 协议 | Google 于 2025 年 4 月推出的开放标准，用于智能体间的发现、认证和通信 |
| Agent Card | Agent 卡片 | A2A 协议中 JSON 格式的智能体能力描述文件 |
| AutoGen (AG2) | AutoGen | Microsoft Research 的多智能体框架，支持对话式 GroupChat 和异步事件驱动 |
| Alignment | 对齐 | 确保 AI 系统的行为符合人类意图和价值观的过程 |
| Abliteration | 安全训练剥离 | 从开源模型中移除 safety alignment 训练的技术 |

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
| Constitutional AI (CAI) | Constitutional AI / 宪法 AI | Anthropic 提出的方法，模型根据一套伦理原则自我审视输出 |
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
| Channel Adapter | 渠道适配器 | 将 Telegram、Discord、Signal 等平台的原生消息转换为统一事件格式的组件 |
| Coordination Drift | 协调漂移 | 多智能体系统中 agent 逐渐失去共同理解的现象 |
| CrewAI | CrewAI | 基于角色的多智能体框架，以快速原型开发著称 |
| Capability Contract | 能力合约 | 对 tool 的名称、参数、权限、执行环境、结果和错误形态的结构化约定 |
| Capability Control | 能力控制 | 根据 tool 的 blast radius 分配权限、approval、sandbox 和审计强度 |

## D

| English | 中文 | 简要说明 |
|---------|------|----------|
| Decoder | 解码器 | Transformer 的生成部分 |
| Dense Embedding | 稠密嵌入 | 低维连续向量表示，每个维度都有值，与稀疏的 one-hot 相对 |
| Distributional Hypothesis | 分布式假设 | 语言学假说：出现在相似上下文中的词具有相似含义 |
| Distillation | 知识蒸馏 | 用大模型训练小模型 |
| Dark Knowledge | 暗知识 | 老师模型对错误答案的概率分布中隐藏的有用信息 |
| On-Policy Distillation | 在线策略蒸馏 | 学生自己生成输出，老师对学生输出提供反馈的蒸馏方法 |
| Defense-in-Depth | 纵深防御 | 多层重叠安全防御策略，每层捕获不同类型的攻击 |
| Dropout | Dropout | 随机丢弃神经元的正则化技术 |

## E

| English | 中文 | 简要说明 |
|---------|------|----------|
| Embedding | 嵌入/向量表示 | 将离散 token 映射到连续向量空间 |
| Emergent Abilities | 涌现能力 | 在特定模型规模突然出现的能力，小模型完全不具备 |
| Emergence | 涌现 | 大模型展现出小模型没有的能力 |
| Encoder | 编码器 | Transformer 的理解部分 |
| Excessive Agency | 过度授权 | OWASP LLM06:2025，给 Agent 超出其需要的权限的风险 |
| Exposure Bias | 暴露偏差 | 训练时看真实前缀、推理时看模型自生成前缀所带来的分布差异 |
| Error Cascade | 错误级联 | 多智能体系统中一个 agent 的错误向下游 agent 传播放大 |

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
| GAIA | GAIA | General AI Assistants Benchmark，Meta 等提出的通用助手评估基准，测试多步推理和工具使用 |
| Gateway Router | 网关路由器 | 在多渠道 agent 系统中统一处理身份、权限、session、delivery route 和执行队列的路由层 |
| Ground Truth | 金标答案 / 标准答案 | 评估中作为正确答案参考的标准输出 |

## H

| English | 中文 | 简要说明 |
|---------|------|----------|
| Hallucination | 幻觉 | LLM 生成虚假但看起来合理的内容 |
| HNSW (Hierarchical Navigable Small World) | 分层可导航小世界图 | 向量数据库中主流的 ANN 索引算法 |
| Hybrid Retrieval | 混合检索 | 结合关键词检索、向量检索和 reranking 的检索策略，常用于企业文档问答 |
| Hidden Layer | 隐藏层 | 神经网络的中间层 |
| Harness (Evaluation Harness) | 评估脚手架 | 包装基础模型并配置工具、prompt、重试策略等，以 Agent 模式运行评估的框架 |

## I

| English | 中文 | 简要说明 |
|---------|------|----------|
| In-context Learning | 上下文学习 | 通过 prompt 中的示例学习，无需更新参数 |
| Inference | 推理 | 使用训练好的模型进行预测 |
| Inference-Time Scaling | 推理时缩放 | 通过增加推理计算（如更长的思维链）提升性能 |
| Idempotency | 幂等性 | 同一事件被重复投递时只触发一次实际动作的可靠性属性 |
| Indirect Prompt Injection | 间接 Prompt 注入 | 通过 Agent 摄入的外部数据（邮件、网页等）隐藏恶意指令的攻击 |
| Instruction Tuning | 指令微调 | 用指令-回答数据训练模型，使其更好地遵循自然语言指令 |

## J

| English | 中文 | 简要说明 |
|---------|------|----------|
| Jailbreak | 越狱 / 绕过安全 | 通过创造性提示绕过 LLM 安全训练的技术 |

## K

| English | 中文 | 简要说明 |
|---------|------|----------|
| KV Cache | KV 缓存 | 在推理时缓存历史 token 的 key/value 张量，避免重复计算 |
| Knowledge Graph | 知识图谱 | 存储实体和关系的图结构数据库，用于关系推理和时间感知记忆 |
| Knowledge Management | 知识管理 | 捕获、组织、检索、验证、治理和更新组织知识的系统性过程 |

## L

| English | 中文 | 简要说明 |
|---------|------|----------|
| Large Language Model (LLM) | 大语言模型 | 大规模预训练的语言模型 |
| Learning Rate | 学习率 | 梯度下降的步长 |
| LoRA | LoRA | Low-Rank Adaptation，参数高效微调方法 |
| Logits | Logits / 未归一化分数 | Softmax 之前的原始输出分数，用于计算概率分布 |
| Loss Function | 损失函数 | 衡量预测与真实值差距的函数 |
| LangGraph | LangGraph | LangChain 团队开发的有向图多智能体框架，以生产级可靠性著称 |

## M

| English | 中文 | 简要说明 |
|---------|------|----------|
| Measurement Artifact | 测量假象 | 由离散指标（如准确率）导致的表面涌现现象 |
| MLP | 多层感知机 | Multi-Layer Perceptron，基础神经网络 |
| Message Routing | 消息路由 | 将不同渠道输入映射到正确身份、session、权限策略和回复路径的系统设计层 |
| Model-Based Reinforcement Learning | 基于模型的强化学习 | 先学习或利用环境模型，再在模型中规划或生成训练信号的强化学习范式 |
| Model-Predictive Control (MPC) | 模型预测控制 | 先用模型评估多条未来动作序列，只执行当前最优方案的前几步，再重新规划 |
| MoE | 混合专家 | Mixture of Experts，稀疏激活架构 |
| Multi-head Attention | 多头注意力 | 并行计算多组注意力 |
| MRL (Matryoshka Representation Learning) | 套娃表示学习 | 可截断嵌入维度的训练方法，降低存储成本 |
| Memory System | 记忆系统 | 智能体的信息持久化与检索机制，包括工作记忆、短期记忆和长期记忆 |
| Multi-Query Attention (MQA) | 多查询注意力 | 多个 query 头共享同一组 key/value 的注意力变体，降低解码内存带宽压力 |
| MAST (Multi-Agent System Failure Taxonomy) | MAST 失败分类法 | NeurIPS 2025 提出的多智能体失败模式分类，含 14 种失败模式 |
| MCP (Model Context Protocol) | 模型上下文协议 | Anthropic 于 2024 年 11 月推出的开放标准，定义 AI 应用连接外部工具和数据的统一协议 |
| MCP Host | MCP 宿主 | 用户交互的 AI 应用，如 Claude Desktop、VS Code，内含 LLM 和 MCP Client |
| MCP Client | MCP 客户端 | 宿主内部的协议处理器，管理与 MCP Server 的 JSON-RPC 通信 |
| MCP Server | MCP 服务器 | 暴露 Tools、Resources、Prompts 三种原语给 AI 应用的外部服务 |
| MCP Registry | MCP 注册中心 | 发现可用 MCP 服务器的中心索引，由社区维护 |
| MCP Apps | MCP 应用 | MCP 的官方扩展，支持在聊天窗口内交付交互式 UI 组件 |
| METR | METR | Model Evaluation and Threat Research，非营利研究组织，提出 Agent 时间窗口评估框架 |
| AAIF (Agentic AI Foundation) | 智能体 AI 基金会 | Linux Foundation 下管理 MCP 的机构，由 Anthropic、Block、OpenAI 共同发起 |
| SEP (Specification Enhancement Proposal) | 规范增强提案 | 社区推动 MCP 协议演进的正式提案机制 |
| Streamable HTTP | 可流式 HTTP | MCP 的远程传输方式，通过 HTTP POST + SSE 实现流式响应 |
| JSON-RPC 2.0 | JSON-RPC 2.0 | MCP 所基于的 JSON 远程过程调用协议标准 |

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
| OSWorld | OSWorld | 在真实操作系统上测试多模态桌面任务 Agent 的基准测试 |

## P

| English | 中文 | 简要说明 |
|---------|------|----------|
| Paged Attention | 分页注意力 | 将 KV 缓存按块存储和映射的推理优化技术，常用于减少内存碎片 |
| Parameter | 参数 | 模型中可学习的权重 |
| Phase Transition | 相变 | 在临界阈值发生的质的变化（物理学概念，用于类比涌现） |
| Pipeline Parallelism | 流水线并行 | 不同 Transformer 层分布在不同 GPU 上，数据像流水线一样流过 |
| pass@k | pass@k | 在 k 次尝试中至少成功一次的概率（乐观可靠性指标） |
| pass^k | pass^k | 在 k 次尝试中全部成功的概率（严格可靠性指标） |
| Position Encoding | 位置编码 | 让模型感知 token 顺序 |
| Position Interpolation | 位置插值 | 通过重缩放位置索引，把更长序列压缩映射到模型更熟悉的位置范围 |
| Power Law | 幂律 | 变量间的乘幂关系，如 $y \propto x^{\alpha}$，缩放定律的数学基础 |
| Pre-training | 预训练 | 在大规模数据上的初始训练 |
| Prefix Caching | 前缀缓存 | 复用共享 prompt 前缀的 prefill 结果，避免重复构建 KV 缓存 |
| Prompt | 提示词 | 给模型的输入指令 |
| Prompt Engineering | 提示工程 | 通过设计指令、上下文、示例和输出约束来改善模型行为的方法 |
| Prompt Injection | Prompt 注入 | 通过将恶意指令与合法数据混合来欺骗 LLM 的攻击技术 |

## Q

| English | 中文 | 简要说明 |
|---------|------|----------|
| Quantization | 量化 | 降低模型精度以减少内存和计算 |

## R

| English | 中文 | 简要说明 |
|---------|------|----------|
| RAG | 检索增强生成 | Retrieval-Augmented Generation，先从外部知识库检索相关文档，再结合 LLM 生成回答的架构 |
| RAGAS | RAGAS 评估框架 | Retrieval Augmented Generation Assessment，评估 RAG 系统忠实度、相关性、召回率等指标的框架 |
| Re-ranking | 重排 | 对初筛结果用更精细的模型（如交叉编码器）重新排序，提升检索质量 |
| Reciprocal Rank Fusion (RRF) | 倒数排名融合 | 将多个检索系统的排名结果融合为统一排序的方法 |
| CRAG | 纠正性 RAG | Corrective RAG，添加评估器判断检索质量，失败时触发后备搜索 |
| Chunking | 分块 | 将长文档切分为较短的文本段落，用于检索和嵌入 |
| ColPali | ColPali | 用视觉 Transformer 直接处理页面图像进行文档检索的方法 |
| Contextual Retrieval | 上下文检索 | Anthropic 提出的在嵌入前为每个文本块添加 LLM 生成上下文的技术 |
| Late Chunking | Late Chunking | 先嵌入整篇文档再分块的技术，保留跨块语义连贯性 |
| GraphRAG | GraphRAG | 微软提出的基于知识图谱的检索增强生成方法 |
| Self-RAG | Self-RAG | 让模型学习自我反思 token，自适应决定是否检索和验证的 RAG 变体 |
| ReLU | ReLU | Rectified Linear Unit，常用激活函数 |
| RLHF | 人类反馈强化学习 | Reinforcement Learning from Human Feedback |
| RNN | 循环神经网络 | Recurrent Neural Network |
| Red Teaming | 红队测试 | 系统性地对 AI 系统进行对抗性压力测试，发现安全漏洞 |
| ReasAlign | ReasAlign | 推理增强的 safety alignment 方法，教会模型检测指令冲突（Li et al., 2026） |
| RLAIF | AI 反馈强化学习 | Reinforcement Learning from AI Feedback，用 AI 生成反馈替代人类反馈 |
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
| Session Docking | Session docking / 会话停靠 | 在保持同一 session 上下文的同时切换回复渠道或 delivery route 的机制 |
| SentencePiece | SentencePiece | Google 的语言无关子词分词器 |
| Skip-gram | Skip-gram | Word2Vec 的一种变体，从中心词预测上下文 |
| Sliding-Window Attention | 滑动窗口注意力 | 每个 token 主要关注固定邻域内的 token，以降低长序列计算和内存成本 |
| Softmax | Softmax | 将 logits 转换为概率分布 |
| SAT Solver | 可满足性求解器 | 用于求解布尔可满足性问题的程序，常用于形式约束检查和组合优化 |
| Subword Tokenization | 子词分词 | 介于字符和词之间的分词策略 |
| SWE-bench | SWE-bench | Princeton 提出的软件工程基准测试，在真实 GitHub issue 上评估 Agent 的 bug 修复能力 |
| Scaffold | 脚手架 | 包装基础模型并配置工具、prompt、重试策略等以 Agent 模式运行的评估/部署框架 |
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
| τ-bench (tau-bench) | τ-bench | Sierra Research 提出的 Agent 可靠性基准测试，通过 pass^k 指标衡量策略遵循的一致性 |
| Time Horizon | 时间窗口 | METR 提出的概念，衡量 Agent 在失败前能自主工作多长时间 |

## T

| English | 中文 | 简要说明 |
|---------|------|----------|
| Tool Schema | Tool Schema / 工具模式 | 描述 tool 输入输出、必填字段、类型、风险元数据和错误形态的结构化契约 |
| Tool System | Tool System / 工具系统 | 管理 agent capabilities 的发现、schema、policy、执行和 trace 的系统层 |
| Tool Registry | Tool Registry / 工具注册表 | 存放可用 tools、版本、发布者、权限范围和运行位置的发现层 |

## U

| English | 中文 | 简要说明 |
|---------|------|----------|
| Universal Approximation | 通用逼近 | 神经网络能逼近任意连续函数 |

## V

| English | 中文 | 简要说明 |
|---------|------|----------|
| Vanishing Gradient | 梯度消失 | 深层网络中梯度趋近于零的问题 |
| Vector Database | 向量数据库 | 存储和检索高维嵌入向量的专用数据库，如 Pinecone、Milvus、pgvector |

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
| Compound Reliability | 复合可靠性 | 多步骤流水线中每步错误相乘导致的端到端成功率下降 |
| Error Cascading | 错误级联 | 一步的失败传播并恶化所有后续步骤的现象 |
| Context Drift | 上下文漂移 | Agent 在长时间运行中逐渐偏离原始指令或目标的现象 |
| WebArena | WebArena | CMU 提出的网页导航基准测试，在真实浏览器环境中评估 Agent 的自主浏览能力 |
| Guardrails | 护栏 | LLM 调用前后的输入/输出验证层，用于拦截不合规内容 |
| Reflexion | 反思模式 | Agent 生成自我批评语言反馈并重试的自纠错模式（Shinn 等人，NeurIPS 2023） |
| Process Reward Model (PRM) | 过程奖励模型 | 对 Agent 推理的每个中间步骤（而非仅最终输出）打分的专用模型 |
| Checkpoint & Recovery | 检查点与恢复 | 在 Agent 执行过程中保存状态以便从已知好点恢复的工程模式 |
| Rate Limit | 速率限制 | API 提供商对单位时间内请求数量的上限，是 Agent 生产环境首要失败原因 |
| Centaur Mode | Centaur 模式 | 人机协作模式之一：人类与 AI 之间有明确的任务分工（源自自由式国际象棋） |
| Cyborg Mode | Cyborg 模式 | 人机协作模式之一：人类与 AI 在同一任务上流畅交替，没有明确边界 |
| Supervisor Mode | Supervisor 模式 | 人机协作模式之一：AI 执行任务，人类在关键检查点审批 |
| Jagged Frontier | Jagged Frontier（锯齿边界） | AI 能力的不规则边界：在相邻任务上可能分别表现超群和失败 |
| Human-in-the-Loop (HITL) | Human-in-the-Loop | 系统设计中要求人类在关键环节参与审批或决策的模式 |
| Human-on-the-Loop | Human-on-the-Loop | 人类不直接参与每一步操作，但可以随时监控和干预的模式 |
| Approval Fatigue | 审批疲劳 | 当人类被要求审批过多决策时，注意力下降、开始机械通过的现象 |
| Centaur Evaluation | Centaur Evaluation | 评估人类-AI 团队联合表现的基准框架（斯坦福 Digital Economy Lab） |
| Autonomy Spectrum | 自主权光谱 | 从完全人类控制到完全 AI 自主的五个渐进级别 |
| Leveling Effect | 拉平效应 | AI 对经验不足者的帮助显著大于经验丰富者的现象 |

---

*持续更新中...*

| MTok | 百万 token | Million tokens，LLM API 定价的计费单位 |
| Prompt Caching | 提示缓存 | 对重复的上下文 token 进行缓存，降低成本和延迟 |
| TTFT | 首 Token 时间 | Time to First Token，从发送请求到收到第一个 token 的延迟 |
| Batch API | 批量 API | 异步处理非紧急请求的接口，通常提供 50% 折扣 |
| LiteLLM | LiteLLM | 开源的统一 LLM 接口库，支持 100+ 提供商 |
| OpenRouter | OpenRouter | 跨提供商 API 网关，支持按成本/延迟/能力路由 |
| Vendor Lock-in | 供应商锁定 | 对单一提供商的技术依赖，增加迁移成本和风险 |
| Responses API | Responses API | OpenAI 2025 年推出的新一代 API，支持内置工具调用和状态管理 |
| MCP | 模型上下文协议 | Model Context Protocol，Anthropic 推出的标准化工具/数据源连接协议 |
| GGUF | GGUF | GPT-Generated Unified Format，llama.cpp 社区开发的模型打包格式，支持多种量化等级和 CPU-GPU 混合推理 |
| AWQ | AWQ | Activation-aware Weight Quantization，激活感知的 4-bit 权重量化方法，质量损失约 1% |
| GPTQ | GPTQ | Generative Pre-trained Transformer Quantization，使用近似二阶信息的训练后量化方法 |
| BitNet | BitNet | Microsoft Research 提出的原生 1.58-bit 训练方法（三值权重 -1/0/+1），通过 BitNet.cpp 实现高效 CPU 推理 |
| Ollama | Ollama | 本地 LLM 部署工具，封装 llama.cpp/MLX，提供 OpenAI 兼容 API，自动选择最佳量化 |
| vLLM | vLLM | 高吞吐 LLM 推理服务引擎，核心创新 PagedAttention，适合生产级 GPU 服务 |
| SGLang | SGLang | 高性能 LLM 服务引擎，通过 RadixAttention 实现 KV cache 前缀复用，2026 年吞吐量领跑 |
| MLX | MLX | Apple 开发的开源数组框架，专为 Apple Silicon 统一内存架构优化 |
| FP8 | FP8 | 8-bit 浮点格式，在 H100/Blackwell 等 GPU 上原生支持，推理时内存减半 |
| RadixAttention | RadixAttention | SGLang 的核心优化，用基数树结构缓存和复用 KV cache 前缀 |
| PagedAttention | 分页注意力 | 将 KV 缓存按块存储和映射的推理优化技术，常用于减少内存碎片 |
| Continuous Batching | 连续批处理 | 动态将新请求加入当前批次的技术，最大化 GPU 利用率 |
| Tensor Parallelism | 张量并行 | 将模型层切分到多个 GPU 上并行计算的技术 |
| Golden Dataset | Golden Dataset（评估数据集） | 精心策划的输入-输出对集合，作为 LLM 应用的评估基准 |
| LLM-as-a-Judge | LLM-as-a-Judge | 用一个 LLM 来评估另一个 LLM 系统输出质量的技术 |
| RAG Triad | RAG 三元组 | RAG 评估的三个核心指标：Faithfulness、Context Relevance、Answer Relevance |
| Evaluation-as-Observability | Evaluation-as-Observability | 将部署前评估与生产监控融合为一条管线的趋势 |
| Pass@k | Pass@k | 运行任务 k 次，至少成功一次即通过的概率指标 |
| Position Bias | Position Bias（位置偏差） | LLM 评审倾向于选择先出现的答案的系统性偏差 |
| OTLP | OTLP | OpenTelemetry Protocol，供应商中立的遥测数据传输协议 |
| OpenInference | OpenInference | 基于 OpenTelemetry 的 LLM 追踪语义约定 |
| Faithfulness | Faithfulness（忠实度） | 衡量 LLM 输出是否基于提供的上下文而非编造的指标 |
| Context Relevance | Context Relevance（上下文相关性） | 衡量检索到的上下文是否与查询相关的指标 |
| Answer Relevance | Answer Relevance（答案相关性） | 衡量 LLM 回答是否真正切题的指标 |
| Drift Detection | Drift Detection（漂移检测） | 监控 LLM 输出质量随时间变化的自动化方法 |
| SWE-bench | SWE-bench | 评估编程 Agent 能否解决真实 GitHub issue 的基准测试 |
| G-Eval | G-Eval | 使用概率加权输出的 LLM 评估方法，由 Liu et al. (2023) 提出 |
| CSAT (Customer Satisfaction Score) | 客户满意度评分 | 客服对话后由客户给出的满意度评分，通常为 1–5 分 |
| FCR (First Contact Resolution) | 首次联系解决率 | 客户首次联系即解决问题的比例，客服核心 KPI |
| STT (Speech-to-Text) | 语音转文字 | 将语音信号转换为文本的技术，如 Whisper、Deepgram |
| TTS (Text-to-Speech) | 文字转语音 | 将文本转换为自然语音的技术，如 ElevenLabs、OpenAI TTS |
| SIP (Session Initiation Protocol) | 会话发起协议 | 用于建立、修改和终止 VoIP 电话会话的标准协议 |
| Resolution Rate | 解决率 | AI Agent 无需人工介入即可完全解决的对话比例 |
| Escalation | 转交 / 升级 | 当 AI 无法处理时，将对话连同上下文转交给人工坐席 |
| Handle Time | 处理时间 | 从客户发起联系到问题解决的总耗时 |
| Deflection | 挡回 / 转向自助 | 将客户从人工渠道引导至 AI 或自助服务的行为 |
| Vibe Coding | Vibe coding | 用自然语言描述意图并让 AI 生成、修改和运行代码的软件创建方式，适合快速原型但需要验证 |
| Coding Agent | 编程 Agent | 能读取代码库、编辑文件、运行工具、观察结果并迭代的软件工程 Agent |
| Agent Harness | Agent Harness | 连接模型、上下文、工具、sandbox、diff 和反馈信号的执行框架 |
| Private Eval | 私有评估集 | 团队针对自身代码库、业务风险和生产约束构建的 Agent 评估集 |
## Day 55: Research and Science

- **AI for Science**: AI systems that accelerate scientific discovery by modeling data, hypotheses, experiments, and feedback loops.
- **Closed-loop discovery**: An iterative workflow that generates candidates, predicts outcomes, selects experiments, tests them, and updates the model with evidence.
- **Active learning**: A strategy for choosing the next data point or experiment based on expected information value, uncertainty, and cost.
- **Acquisition function**: A scoring rule used in active learning or Bayesian optimization to decide which candidate should be tested next.
- **AI scientist**: An agentic system that automates parts of scientific workflow such as ideation, literature search, experiment execution, analysis, and writing.

## Day 56: OpenClaw Architecture Overview

- **Gateway**: The long-lived OpenClaw control plane that owns routing, sessions, channel connections, and agent run coordination.
- **Channel adapter**: A platform adapter that normalizes messages from surfaces such as Telegram, Slack, Discord, WhatsApp, WebChat, cron, or CLI into common events.
- **Session key**: A routing/context selector used to choose the conversation state for a message; it is not an authorization token.
- **Channel docking**: Moving the delivery route of an existing OpenClaw session to another linked channel while keeping the same transcript and context.
- **Trust boundary**: The line that separates who is trusted to cause actions, which credentials/tools they can use, and which host or gateway should enforce that separation.

## Day 59: Memory and Context Management

- **Memory management**: The process of deciding what durable state an agent stores, updates, expires, retrieves, and deletes across turns and sessions.
- **Context management**: The process of selecting and organizing the limited information that enters the model context window for one inference.
- **Compaction**: Compressing long interaction history into a smaller task state while preserving goals, constraints, decisions, open questions, and provenance.
- **Episodic memory**: Time-stamped memory of events, decisions, outcomes, and source traces.
- **Semantic memory**: Stable facts, preferences, and domain knowledge used across tasks.
- **Procedural memory**: Reusable rules, skills, workflows, policies, and conventions that guide behavior.
- **Provenance**: Metadata that records where a memory or claim came from so it can be audited, corrected, or deleted.
