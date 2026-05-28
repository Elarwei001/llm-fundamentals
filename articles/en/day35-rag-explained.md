# Day 35: RAG Explained — Retrieval-Augmented Generation

> **Core Question**: How do you give an LLM access to knowledge it wasn't trained on, without retraining it?

---

## Opening

Imagine you're a brilliant professor who has memorized every textbook published before 2023. You can answer almost any question off the top of your head. But then someone asks you about a paper published yesterday. You could guess — confidently — but you'd probably be wrong.

That's exactly the situation LLMs find themselves in. They're frozen in time at their training cutoff, with no access to private documents, recent news, or your company's internal wiki. RAG (Retrieval-Augmented Generation) is the solution: instead of memorizing everything, the model *looks things up* before answering.

RAG was introduced by Patrick Lewis and colleagues at Meta AI (then Facebook AI Research), University College London, and NYU in their landmark 2020 NeurIPS paper ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401). The core idea was elegant: combine a pre-trained parametric model (the LLM) with a non-parametric memory (a retrieval index). What started as a research idea has since become one of the most widely deployed LLM architectures in production — from enterprise chatbots to coding assistants.

---

## 1. Why RAG Exists: The Knowledge Problem

#### Intuition: The Open-Book Exam

Think of a standard LLM as a student taking a *closed-book exam*. It can only use what it memorized during training. RAG turns it into an *open-book exam* — the model can look up relevant passages before crafting its answer. The key insight: you don't need to memorize the entire library if you're good at finding the right page.

LLMs face several knowledge limitations that RAG directly addresses:

| Limitation | Description | RAG's Answer |
|-----------|-------------|--------------|
| **Knowledge cutoff** | Training data has a fixed end date | Retrieve from up-to-date sources |
| **Private data** | Corporate wikis, medical records, legal docs | Index and retrieve from internal documents |
| **Hallucination** | Model fabricates plausible but false facts | Ground generation in retrieved evidence |
| **Domain specificity** | General knowledge, weak on niche topics | Specialized corpora boost domain accuracy |
| **Cost of updates** | Retraining is expensive and slow | Update the retrieval index in minutes |

---

## 2. How RAG Works: The Full Pipeline

The RAG pipeline has two major phases: an **offline indexing phase** (done once) and an **online query phase** (done per user question).

### 2.1 Offline Indexing

Before any questions arrive, you prepare your knowledge base:

1. **Document Collection** — Gather your sources: PDFs, wiki pages, support tickets, code repositories.
2. **Chunking** — Split documents into smaller passages (typically 256–512 tokens). This is critical because feeding entire documents into the retrieval step is both expensive and imprecise.
3. **Embedding** — Pass each chunk through an embedding model (e.g., [text-embedding-3-large](https://platform.openai.com/docs/guides/embeddings) from OpenAI, or open-source alternatives like [BGE](https://huggingface.co/BAAI/bge-large-en-v1.5)) to produce a dense vector representation.
4. **Indexing** — Store these vectors in a vector database such as [Pinecone](https://www.pinecone.io/), [Weaviate](https://weaviate.io/), [Chroma](https://www.trychroma.com/), or [Qdrant](https://qdrant.tech/), optimized for fast nearest-neighbor search.

### 2.2 Online Query

When a user asks a question:

1. **Query Encoding** — The user's question is embedded using the same embedding model.
2. **Retrieval** — The query vector is compared against all chunk vectors in the database. The top-k most similar chunks are retrieved (typically k=3 to k=10).
3. **Prompt Construction** — The retrieved chunks are injected into the LLM's prompt, usually with a template like: "Use the following context to answer the question. Context: [chunks]. Question: [query]."
4. **Generation** — The LLM reads the prompt (with retrieved context) and generates a grounded answer.

![RAG Pipeline Architecture](../zh/images/day35/rag-pipeline-architecture.png)
*Figure 1: The complete RAG pipeline — from user query through retrieval to final answer generation.*

---

## 3. The Chunking Problem: Where RAG Lives or Dies

#### Intuition: Cutting a Novel Into Random Pages

Imagine taking a 500-page novel and cutting it into 50-page sections at arbitrary points — maybe mid-sentence, mid-paragraph. Then someone asks "Why did the protagonist go to Paris?" and you hand them pages 200–250, which happen to start in the middle of a dinner scene and end mid-chase. The context is fragmented and confusing.

That's exactly what bad chunking does to a RAG system. The chunking strategy is arguably the single most impactful engineering decision in a RAG pipeline, because it determines what the retriever can find.

### 3.1 Chunking Strategies

| Strategy | How It Works | Pros | Cons |
|----------|-------------|------|------|
| **Fixed-size** | Split every N tokens with overlap | Simple, predictable | Cuts mid-sentence, loses semantic coherence |
| **Sentence-based** | Split at sentence boundaries | Clean linguistic units | Chunks may be too small or too large |
| **Recursive** | Split by headers → paragraphs → sentences | Respects document structure | Needs structured input (Markdown, HTML) |
| **Semantic** | Group sentences by embedding similarity | Semantically coherent chunks | More compute, may produce uneven sizes |
| **Late Chunking** | Embed whole doc first, *then* chunk | Preserves cross-chunk context | Requires special embedding models |

### 3.2 The Late Chunking Revolution

A significant advance came from [Jina AI's Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) technique (2024). In traditional RAG, you chunk first, then embed — meaning each chunk's embedding is isolated from the rest of the document. Late Chunking inverts this: embed the *entire document* first, then partition the embedding into chunks. This preserves inter-chunk context and reduces retrieval failures by up to 30% on complex queries.

Similarly, [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (September 2024) prepends a brief, LLM-generated context to each chunk before embedding — explaining where this chunk fits within its parent document. Anthropic reported a 67% reduction in top-20 retrieval failures with this simple technique.

![Chunking and Embedding Process](../zh/images/day35/chunking-and-embedding-process.png)
*Figure 2: The offline indexing pipeline — from raw documents through chunking and embedding to vector storage.*

---

## 4. Retrieval: Finding the Right Needles

#### Intuition: A Library With a Magical Card Catalog

In a physical library, you search the card catalog by title or author — but what if you want "books about resilience in the face of failure"? You'd need to already know the right keywords. Dense retrieval is like having a librarian who understands *meaning*, not just keywords.

### 4.1 Sparse vs. Dense Retrieval

| Aspect | Sparse Retrieval (BM25) | Dense Retrieval (Embeddings) |
|--------|------------------------|------------------------------|
| **Method** | Keyword frequency matching | Semantic vector similarity |
| **Strength** | Exact keyword matches, proper nouns | Conceptual matching, synonyms |
| **Weakness** | Misses paraphrases, synonyms | Weaker on rare terms, names |
| **Speed** | Very fast (inverted index) | Fast (approximate nearest neighbor) |
| **Best for** | Legal docs, code, technical terms | FAQs, conversational queries |

#### How Does BM25 Work?

**BM25** (Best Matching 25) was developed by Stephen Robertson and Karen Spärck Jones in the 1990s and remains a classic in information retrieval. Its intuition: **the more a word appears in a document, the more relevant; but if it appears in almost every document, it's less discriminative.**

The core formula can be simplified as:

```
BM25 score = Σ for each query term:
    IDF(term) × freq(term, doc) × (k1 + 1)
    ─────────────────────────────────────────
    freq(term, doc) + k1 × (1 - b + b × doc_length / avg_doc_length)
```

Three key parameters:

- **IDF (Inverse Document Frequency)**: Measures how rare a word is. "the" appears in nearly every document (IDF ≈ 0); "BM25" is rare (high IDF). Rare words contribute more to matching.
- **Term frequency saturation (k1)**: A word appearing 10 times isn't necessarily 2× better than appearing 5 times. BM25 uses a saturation function for diminishing returns — the first occurrence contributes the most. Typically k1 = 1.2–2.0.
- **Document length normalization (b)**: Longer documents naturally match more words. BM25 penalizes overly long documents via the b parameter. b = 0 means no normalization; b = 1 means full normalization. Typically b = 0.75.

**Example:**

Query: "application scenarios of BM25 algorithm"

- Document A: "BM25 is a classic search algorithm widely applied in e-commerce search" (short, exact match of 2 keywords)
- Document B: "Search engines use multiple algorithms including TF-IDF, BM25, BERT, covering various application scenarios..." (longer, also matches keywords, but penalized for length)

BM25 gives Document A a higher score — keywords are more concentrated and the document is more concise.

**Why is it still used in 2026?**

Despite embedding models and semantic search, BM25 remains a standard component in production RAG systems:
- More reliable for **exact matches** (names, product IDs, code variable names) than embedding models
- **Zero latency overhead** — based on inverted indices, no GPU needed
- **Complementary** to dense retrieval — BM25 catches exact matches, embeddings catch semantic similarity

This is why hybrid retrieval (BM25 + dense retrieval) is the current production standard.

Modern RAG systems typically use **hybrid retrieval** — combining BM25 (sparse) and dense retrieval — to get the best of both worlds. The results are merged using reciprocal rank fusion (RRF) or a learned re-ranker.

### 4.2 Re-ranking: The Quality Filter

Initial retrieval (whether BM25 or dense) has a fundamental problem: **it never looks at the query and document *together* when scoring.**

#### Why Isn't Initial Retrieval Enough?

**Bi-encoders** — the models used for dense retrieval — encode the query and document *separately* into vectors, then compute cosine similarity between them. The problem:

- Query and document are each compressed into a single vector, losing significant detail
- "Apple" in the query might mean the fruit, but in the document it might mean the company — separately encoded, this distinction is lost
- BM25 has a similar issue: it only checks whether keywords appear, without understanding context

Analogy: A bi-encoder is like judging whether two people match by looking at their passport photos separately — fast but crude.

A **re-ranker (Cross-Encoder)** puts both people in the same room and lets them interact before judging — slow but precise.

#### How Does a Cross-Encoder Work?

A cross-encoder concatenates the query and document *together* and feeds them as one input to a Transformer:

```
Input: [CLS] What new products did Apple release recently [SEP] Apple Inc. released the new MacBook Pro in May 2026 [SEP]
                              ↕
                Transformer joint encoding
                              ↕
              Relevance score: 0.94

Input: [CLS] What new products did Apple release recently [SEP] Apples are rich in vitamin C and eating one daily promotes health [SEP]
                              ↕
                Transformer joint encoding
                              ↕
              Relevance score: 0.12
```

Key difference: every token can *attend* to all other tokens in both the query and the document. The model directly sees that "Apple" refers to a company in the query context but a fruit in the second document — no guessing required.

#### Why Two Stages Are Necessary

Why not just use a cross-encoder for everything? Because **it's too slow**.

| | Bi-Encoder (Initial Retrieval) | Cross-Encoder (Re-ranking) |
|---|---|---|
| **Query vs. Document** | Encoded separately, compare vectors | Concatenated, joint understanding |
| **Speed** | Very fast (pre-computed doc vectors) | Slow (run full Transformer per pair) |
| **Suitable scale** | 1M+ documents | 50–100 candidates |
| **Precision** | Medium | High |
| **Compute cost** | Low | High |

If you have 1M documents, running a cross-encoder on each one would take minutes per query — users won't wait. Hence the two-stage strategy:

1. **Stage 1**: Use a fast bi-encoder to retrieve top-50 from 1M documents (milliseconds)
2. **Stage 2**: Use a cross-encoder to precisely score those 50 and keep top-5 (seconds)

#### What Problems Does Re-ranking Solve?

- **Disambiguation**: Is "apple" the company or the fruit? The cross-encoder judges based on full query context
- **Polysemy/Synonymy**: "Cheap" and "affordable" may not be close enough in vector space, but a cross-encoder understands they're synonymous in the current query context
- **Long-tail queries**: Users ask very specific questions where bi-encoders match keywords but miss intent — re-rankers correct this
- **Cross-lingual**: When the query is in Chinese but documents are in English, cross-encoders typically handle cross-lingual semantics better

Popular re-rankers include [Cohere Rerank](https://cohere.com/rerank) and open-source models like [bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large).

---

## 5. RAG Variants: Beyond Vanilla

The original RAG design (retrieve → stuff into prompt → generate) is now called **"Vanilla RAG"** or **"Naive RAG."** As the field matured, researchers identified systematic weaknesses and proposed increasingly sophisticated variants.

![RAG Variants Comparison](../zh/images/day35/rag-variants-comparison.png)
*Figure 3: Four major RAG architecture variants — each addresses a specific weakness of vanilla RAG.*

### 5.1 Self-RAG (Asai et al., 2023)

["Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"](https://arxiv.org/abs/2310.11511) taught the model to ask itself three questions:
- **Should I retrieve?** — Not every question needs external knowledge.
- **Is the retrieved document relevant?** — If not, try again.
- **Is my answer supported by the document?** — If not, revise.

The model learns special *reflection tokens* that trigger these self-checks, making it adaptive rather than always-retrieve.

### 5.2 Corrective RAG (CRAG) (Yan et al., 2024)

["Corrective Retrieval Augmented Generation"](https://arxiv.org/abs/2401.15884) adds a lightweight *evaluator* that scores the quality of retrieved documents. If the documents are irrelevant:
- The system triggers a **web search** fallback to find better information
- If documents are partially relevant, it extracts and refines the useful parts

This "self-healing" retrieval layer is especially valuable in production where your index is never perfectly comprehensive.

### 5.3 GraphRAG (Microsoft, 2024)

Microsoft's [GraphRAG](https://microsoft.github.io/graphrag/) takes a fundamentally different approach to indexing. Instead of chunking documents into flat text passages, it:
1. Uses an LLM to extract entities and relationships from documents
2. Builds a **knowledge graph** connecting entities across documents
3. Detects communities in the graph and generates summaries for each community
4. At query time, retrieves from both the graph structure and community summaries

GraphRAG excels at **global questions** that require synthesizing information across many documents — queries like "What are the main themes in this 10,000-document corpus?" where vanilla RAG struggles because no single chunk contains the full answer.

---

## 6. The Math Behind RAG

The following formulas come from the original RAG paper—["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) by Lewis et al., published at NeurIPS 2020. The core idea is to combine a pre-trained parametric model (LLM) with non-parametric memory (a retrieval index), and the probabilistic framework below is the mathematical expression of that idea. For readers who want to understand the formal formulation, here's how RAG was originally defined:

$$
\begin{aligned}
p(y \mid x) &= \sum_{z \in \mathcal{Z}} p(y \mid x, z) \cdot p(z \mid x)
\end{aligned}
$$

Where:
- **x** is the input query
- **y** is the generated output
- **z** is a retrieved document
- **p(z | x)** is the retrieval probability (how relevant is document z to query x)
- **p(y | x, z)** is the generation probability (the LLM's answer given query + document)

In practice, the summation is approximated by the top-k retrieved documents:

$$
\begin{aligned}
p(y \mid x) &\approx \sum_{i=1}^{k} p(y \mid x, z_i) \cdot p(z_i \mid x)
\end{aligned}
$$

The retrieval score **p(z | x)** is typically computed as cosine similarity in the embedding space:

$$
\begin{aligned}
\text{sim}(q, d) &= \frac{\mathbf{E}(q) \cdot \mathbf{E}(d)}{\|\mathbf{E}(q)\| \cdot \|\mathbf{E}(d)\|}
\end{aligned}
$$

Where **E(q)** and **E(d)** are the embedding vectors of the query and document respectively.

---

## 7. RAG Performance: What the Benchmarks Tell Us

![RAG Performance Charts](../zh/images/day35/rag-performance-charts.png)
*Figure 4: Retrieval failure rates by chunking strategy (left) and accuracy across RAG variants on complex QA (right). Data reflects Meta CRAG Benchmark and Anthropic contextual retrieval evaluations.*

Key takeaways from the data:

1. **Chunking matters enormously** — The gap between naive chunking (39% failure) and contextual retrieval (13% failure) is 26 percentage points — a 3x improvement in retrieval quality.
2. **Advanced variants consistently outperform vanilla RAG** — On Meta's CRAG benchmark, vanilla RAG achieves only 63% accuracy, while agentic RAG reaches 85%.
3. **Diminishing returns with complexity** — Each additional layer of sophistication (re-ranker, self-reflection, correction) adds 4-7% accuracy but also adds latency and infrastructure complexity.

---

## 8. Building RAG in Practice

### 8.1 The Tech Stack

| Component | Popular Choices | Notes |
|-----------|----------------|-------|
| **Embedding Model** | OpenAI text-embedding-3, BGE, E5, Cohere Embed | Match model to your language/domain |
| **Vector Database** | Pinecone, Weaviate, Qdrant, Chroma, pgvector | Cloud-managed vs. self-hosted tradeoff |
| **Framework** | [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/), Haystack | LlamaIndex is RAG-focused; LangChain is more general |
| **LLM** | GPT-4, Claude, Gemini, Llama 3 | Larger models handle long contexts better |
| **Re-ranker** | Cohere Rerank, bge-reranker, ColBERT | Cross-encoder for final quality boost |

### 8.2 Code Example: Minimal RAG Pipeline

```python
# A minimal RAG pipeline using LangChain and OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load and chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # tokens per chunk
    chunk_overlap=50,     # overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # split hierarchy
)
chunks = splitter.split_documents(documents)

# Step 2: Embed and index
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="my_knowledge_base"
)

# Step 3: Retrieve relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke("What is our company's refund policy?")

# Step 4: Generate grounded answer
llm = ChatOpenAI(model="gpt-4o", temperature=0)
context = "\n\n".join(doc.page_content for doc in relevant_docs)

response = llm.invoke(f"""Answer the question based on the context below.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: What is our company's refund policy?""")

print(response.content)
```

This is a basic but functional RAG system in about 25 lines of code. Production systems add re-ranking, hybrid search, query rewriting, evaluation pipelines, and caching — but the core loop remains the same.

---

## 9. Common Misconceptions

### ❌ "RAG eliminates hallucination"

RAG *reduces* hallucination by grounding the model in retrieved evidence, but it doesn't eliminate it. The model can still misinterpret the retrieved text, cherry-pick from conflicting documents, or ignore the context entirely. In adversarial settings, RAG systems can even be manipulated through poisoned documents in the retrieval index.

### ❌ "Bigger context windows make RAG obsolete"

Long-context models (Gemini's 2M tokens, Claude's 200K tokens) can absorb entire documents. But stuffing everything into context is:
- **Expensive** — You pay for every token in the context window
- **Slow** — Attention is quadratic; longer context = slower inference
- **Less precise** — "Lost in the middle" effects mean models may miss relevant information buried in a sea of text

RAG and long contexts are complementary: RAG retrieves the needle, long context gives you a bigger haystack *if you need it*.

### ❌ "Vector search alone is sufficient"

Pure semantic search misses exact keyword matches that matter (product IDs, error codes, proper nouns). Production systems almost always use hybrid retrieval combining both sparse (BM25) and dense (embedding) methods.

---

## 10. The Frontier: Where RAG Is Heading (2025–2026)

The RAG field is moving fast. Here are the most significant recent developments:

### Agentic RAG (2025–2026)

The biggest trend is **Agentic RAG** — where an AI agent decides *when*, *where*, and *how* to retrieve information, rather than following a fixed pipeline. An [Agentic RAG survey by Singh et al. (January 2025)](https://arxiv.org/abs/2501.09136) proposes a taxonomy based on agent cardinality, control structure, and autonomy. In agentic RAG, the LLM can:
- Decide whether retrieval is needed for this query
- Choose between multiple data sources (internal wiki, web search, database query)
- Iteratively refine its search based on intermediate results
- Use tools like SQL queries or API calls alongside document retrieval

### ColPali: Visual Document Retrieval (2025)

[ColPali](https://arxiv.org/abs/2407.01449), introduced by Faysse et al. (July 2024) and refined through 2025, treats document retrieval as a *visual* problem. Instead of extracting text and embedding it, ColPali directly processes page images using a vision transformer with late interaction (ColBERT-style) matching. This is especially powerful for documents with tables, figures, and complex layouts where text extraction loses critical formatting information.

### RAG Evaluation: RAGAS and Beyond

[Evaluating RAG systems](https://docs.ragas.io/) has matured significantly. The RAGAS (Retrieval Augmented Generation Assessment) framework, originally released in 2023 and continuously updated, measures:
- **Faithfulness** — Is the answer grounded in retrieved context?
- **Answer Relevance** — Does the answer address the question?
- **Context Precision** — Are the retrieved chunks actually relevant?
- **Context Recall** — Did retrieval find all necessary information?

A [comprehensive RAG evaluation survey by Chen et al. (January 2026)](https://arxiv.org/abs/2504.14891) found that LLM-based judges are increasingly reliable for RAG evaluation, with 2024 H2 and 2025 H1 seeing the highest publication rates for evaluation methods.

### GraphRAG Matures (2025–2026)

Microsoft's GraphRAG has spawned an entire research sub-field. A [Graph Retrieval-Augmented Generation survey published in ACM TOIS (2025)](https://dl.acm.org/doi/10.1145/3777378) catalogs dozens of graph-enhanced RAG methods. GraphRAG consistently outperforms vanilla RAG on multi-hop reasoning and cross-document synthesis, though at significantly higher indexing cost (100–1000x more expensive to build the index).

![RAG Evolution Timeline](../zh/images/day35/rag-evolution-timeline.png)
*Figure 5: The evolution of RAG from 2020's vanilla retrieve-then-generate to 2026's multimodal agentic systems — retrieval quality, system autonomy, and modality have all advanced significantly.*

---

## 11. Further Reading

### Beginner
1. [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — Hands-on walkthrough of building a RAG system
2. [LlamaIndex Documentation](https://docs.llamaindex.ai/) — Framework specifically designed for RAG applications
3. [Pinecone Learning Center](https://www.pinecone.io/learn/) — Excellent vector search and RAG tutorials

### Advanced
1. [RAGAS Framework Documentation](https://docs.ragas.io/) — Evaluate your RAG pipeline systematically
2. [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) — Graph-enhanced retrieval for complex reasoning
3. [Anthropic Contextual Retrieval Blog Post](https://www.anthropic.com/news/contextual-retrieval) — Simple technique that dramatically improves retrieval

### Re-ranking & Retrieval Refinement
Re-ranking is one of the most impactful yet least discussed components of a RAG pipeline. These resources will help you go deeper:

1. ["Dense Passage Retrieval for Open-Domain Question Answering"](https://arxiv.org/abs/2004.04906) — Karpukhin et al., 2020 (foundational work on bi-encoder retrieval)
2. ["ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"](https://arxiv.org/abs/2004.12832) — Khattab et al., 2020 (late interaction model that balances speed and precision)
3. [Cohere Rerank Documentation](https://docs.cohere.com/docs/reranking) — Practical guide to using a commercial reranking API
4. [bge-reranker Series](https://huggingface.co/BAAI/bge-reranker-v2-m3) — BAAI's open-source reranking models with multilingual support
5. ["A Gentle Introduction to Cross-Encoders for Sentence Pair Scoring"](https://www.sbert.net/examples/applications/cross-encoder/README.html) — Sentence-Transformers docs explaining cross-encoders with intuitive examples

### Papers
1. ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020 (the original RAG paper)
2. ["Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"](https://arxiv.org/abs/2310.11511) — Asai et al., 2023
3. ["Corrective Retrieval Augmented Generation"](https://arxiv.org/abs/2401.15884) — Yan et al., 2024
4. ["Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"](https://arxiv.org/abs/2501.09136) — Singh et al., January 2025
5. ["Retrieval-Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey"](https://arxiv.org/abs/2504.14891) — Chen et al., January 2026
6. ["Graph Retrieval-Augmented Generation: A Survey"](https://dl.acm.org/doi/10.1145/3777378) — ACM TOIS, 2025

---

## Reflection Questions

1. When would you choose RAG over fine-tuning for a domain-specific application? What are the tradeoffs?
2. Why does chunking strategy matter more than the choice of embedding model in most RAG systems?
3. If an agentic RAG system can decide when to retrieve, what happens when it *chooses not to* retrieve but should have? How would you detect and fix this?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| **RAG** | Retrieve relevant documents, inject into LLM prompt, generate grounded answer |
| **Chunking** | Split documents into passages for retrieval; strategy matters enormously |
| **Embedding** | Convert text to dense vectors for semantic similarity search |
| **Vector Database** | Specialized storage for fast nearest-neighbor search on embeddings |
| **Re-ranking** | Two-stage retrieval: fast initial search → expensive quality scoring |
| **Self-RAG** | Model learns to self-reflect: should I retrieve? is it relevant? is my answer supported? |
| **CRAG** | Adds a self-healing evaluator that triggers web search when retrieval fails |
| **GraphRAG** | Builds a knowledge graph from documents for cross-document reasoning |
| **Agentic RAG** | An agent decides when/how to retrieve, choosing between multiple strategies |
| **Contextual Retrieval** | Prepends LLM-generated context to each chunk before embedding |

**Key Takeaway**: RAG solves the fundamental problem of giving LLMs access to knowledge they weren't trained on. The core pipeline is simple — retrieve relevant chunks, inject them into the prompt, generate — but production quality depends heavily on chunking strategy, retrieval method, and increasingly on agentic control. In 2026, the frontier has moved from "can we retrieve?" to "can the system decide *when* and *how* to retrieve on its own?"

---

*Day 35 of 60 | LLM Fundamentals*
*Word count: ~2800 | Reading time: ~14 minutes*
