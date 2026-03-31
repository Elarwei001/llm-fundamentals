# Day 7: Tokenization — The Rosetta Stone of Language Models

> **Core Question**: How do language models convert human text into numbers they can process, and why does this seemingly mundane step profoundly affect everything from model performance to your API bill?

---

## Opening: The Hidden Translation Layer

Imagine trying to teach a calculator to write poetry. The calculator only understands numbers, but poetry is made of words, punctuation, and whitespace. Before any computation can happen, you need a translation system—a way to convert "Shall I compare thee to a summer's day?" into something like `[21421, 314, 8094, 534, 284, 257, 3931, 338, 1110, 30]`.

This translation system is called a **tokenizer**, and it's arguably the most underappreciated component of modern language models. While researchers obsess over attention mechanisms and scaling laws, tokenization quietly determines:

- How much your API call costs (tokens, not words, are billed)
- Whether your model can handle code, math, or non-English text gracefully
- How many words fit in your context window
- Why "ChatGPT can't count" memes exist

In this article, we'll demystify tokenization—from its historical evolution to the elegant algorithms behind modern tokenizers, and the surprising ways it shapes LLM behavior.

![Tokenization Pipeline](../zh/images/day07/tokenization-pipeline.png)
*Figure 1: The tokenization pipeline transforms raw text into numerical vectors that models can process. Each step is crucial but often invisible to users.*

---

## 1. Why Tokenization Matters: Beyond Simple Word Splitting

At first glance, tokenization seems trivial: just split text by spaces, right? The word "hello" becomes one token, "world" becomes another. Problem solved.

But this naive approach immediately runs into trouble:

### The Out-of-Vocabulary (OOV) Problem

If your vocabulary only contains words seen during training, what happens when a user types "ChatGPT" or "cryptocurrency" or even a misspelled "teh"? With word-level tokenization, these become `[UNK]` (unknown) tokens—black boxes that destroy meaning.

Consider: "The new GPT-4o model is impressive" might become "The new [UNK] model is impressive"—losing the most important information in the sentence.

### The Vocabulary Size Explosion

English alone has over 170,000 words in current use, plus millions of proper nouns, technical terms, and slang. Add other languages, code, and emoji, and you're looking at millions of potential tokens. Each token needs an embedding vector (typically 768-4096 dimensions), so vocabulary size directly impacts:

$$
\text{Embedding Memory} = \text{Vocab Size} \times \text{Embedding Dim} \times 4 \text{ bytes}
$$

A vocabulary of 1 million tokens with 4096-dimensional embeddings would require **16 GB** just for the embedding table!

### The Sequence Length Problem

On the flip side, character-level tokenization (where each character is a token) solves OOV completely—any text can be represented. But "hello world" becomes 11 tokens instead of 2, dramatically increasing sequence length. Since Transformer attention is $O(n^2)$ in sequence length, this is computationally disastrous.

**The fundamental tension**: We need a vocabulary small enough to be practical, yet expressive enough to handle any text efficiently.

![Tokenization Strategies](../zh/images/day07/tokenization-strategies.png)
*Figure 2: Character-level tokenization creates very long sequences, word-level requires enormous vocabularies, but subword tokenization (like BPE—Byte Pair Encoding) hits the sweet spot.*

---

## 2. Subword Tokenization: The Elegant Solution

The breakthrough insight behind modern tokenizers is **subword tokenization**: instead of fixed words or characters, learn a vocabulary of variable-length pieces that balance coverage and efficiency.

Common words like "the" or "and" get their own tokens (efficient), while rare words like "tokenization" are split into pieces like `["token", "ization"]` (coverage). This way:

- **No OOV**: Any word can be constructed from subword pieces
- **Moderate vocabulary**: Typically 30K-100K tokens
- **Reasonable sequences**: Common text stays compact

Three algorithms dominate modern NLP:

### 2.1 Byte Pair Encoding (BPE)

Originally a data compression algorithm from 1994, BPE was adapted for NLP in 2016. The idea is beautifully simple:

1. **Start with characters**: Initialize vocabulary with all unique characters
2. **Count pairs**: Find the most frequent adjacent pair
3. **Merge**: Combine that pair into a new token
4. **Repeat**: Continue until reaching desired vocabulary size

![BPE Algorithm](../zh/images/day07/bpe-algorithm-v2.png)
*Figure 3: BPE iteratively merges the most frequent character pairs. After many iterations, common words become single tokens while rare words remain split.*

Let's trace through a concrete example with corpus: `["low", "lower", "newest", "widest"]`

**Iteration 0** (character initialization):
```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, </w>}
Corpus: l o w </w>, l o w e r </w>, n e w e s t </w>, w i d e s t </w>
```

**Iteration 1**: Most frequent pair is `(e, s)` with count 2
```
New token: es
Corpus: l o w </w>, l o w e r </w>, n e w es t </w>, w i d es t </w>
```

**Iteration 2**: Most frequent pair is `(es, t)` with count 2
```
New token: est
Corpus: l o w </w>, l o w e r </w>, n e w est </w>, w i d est </w>
```

After many more iterations, you get tokens like `low`, `er`, `est`, `new`, `wid`—capturing meaningful morphemes.

**Key property**: BPE is **deterministic**. Given the same corpus and merge count, you always get the same vocabulary. This is crucial for reproducibility.

### 2.2 WordPiece

Developed by Google for BERT, WordPiece is similar to BPE but uses a different merge criterion:

$$
\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}
$$

Instead of just frequency, WordPiece maximizes the **likelihood** of the training data. It merges pairs that appear together more often than expected by chance.

#### The Math Behind WordPiece

**What does "maximize likelihood" mean?**

Given a vocabulary $V$, the likelihood of tokenizing a corpus is:

$$
P(\text{corpus} | V) = \prod_{\text{word } w} P(w | V)
$$

For each word, we compute the probability of its tokenization. If "tokenization" is split into `["token", "ization"]`:

$$
P(\text{"tokenization"}) = P(\text{token}) \times P(\text{ization})
$$

where each subword probability is estimated from corpus frequency:

$$
P(\text{subword}) = \frac{\text{count}(\text{subword})}{\text{total tokens}}
$$

**Why the score formula works:**

The WordPiece score $\frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}$ is actually the **Pointwise Mutual Information (PMI)**—it measures how much more likely $x$ and $y$ appear together versus by chance.

- **PMI > 1**: $x$ and $y$ appear together more than expected → good merge candidate
- **PMI = 1**: Independent, appearing together by chance
- **PMI < 1**: $x$ and $y$ avoid each other

**Example**: If "to" appears 1000 times, "ken" appears 500 times, and "token" appears 400 times in a corpus of 10,000 words:
- Expected co-occurrence by chance: $\frac{1000 \times 500}{10000} = 50$
- Actual co-occurrence: 400
- PMI-like score: $\frac{400}{50} = 8$ → Strong signal to merge!

**BPE vs WordPiece:**
- **BPE**: Merge most frequent pair → greedy, simple
- **WordPiece**: Merge pair with highest PMI → statistically principled

In practice, both produce similar results, but WordPiece tends to create more linguistically meaningful subwords.

**Visual marker**: WordPiece uses `##` to indicate continuation. "tokenization" becomes `["token", "##ization"]`. The `##` signals "this piece continues the previous token."

### 2.3 SentencePiece

Developed by Google for multilingual models, SentencePiece treats the input as a raw byte stream—no pre-tokenization by whitespace. This makes it truly **language-agnostic**: Chinese, Japanese, and emoji work just as well as English.

SentencePiece can implement either BPE or a variant called Unigram (which starts with a large vocabulary and prunes down). It's used by T5, LLaMA, and Gemma.

**Key innovation**: By treating whitespace as a regular character (often marked as `▁`), SentencePiece handles languages without spaces naturally.

#### Clarification: Algorithm vs. Tool

It's easy to confuse these terms. Here's the key distinction:

| | BPE / WordPiece | SentencePiece |
|--|-----------------|---------------|
| **What it is** | Tokenization **algorithm** | Tokenization **tool/library** |
| **Level** | Merge strategy (how to pick pairs) | Implementation framework (how to process input) |
| **Analogy** | Recipe (the method) | Kitchen (the equipment) |

**SentencePiece is a tool that can use different algorithms internally:**

```
SentencePiece (tool) + BPE (algorithm)     → LLaMA, T5
SentencePiece (tool) + Unigram (algorithm) → Also supported
WordPiece (algorithm) + custom impl        → BERT
```

When you see "LLaMA uses BPE (SentencePiece)", it means: **SentencePiece library implementing the BPE algorithm**.

---

## 3. Vocabulary Size: The Goldilocks Problem

Choosing vocabulary size is a critical design decision:

| Vocab Size | Pros | Cons |
|------------|------|------|
| Small (8K) | Low memory, fast embedding lookup | Long sequences, poor rare-word handling |
| Medium (32K) | Good balance for English | May struggle with multilingual |
| Large (100K+) | Short sequences, better coverage | High memory, slower softmax |

![Vocabulary Size Trade-off](../zh/images/day07/vocab-size-tradeoff.png)
*Figure 4: Vocabulary size involves a fundamental trade-off between sequence length and memory/computation. Modern LLMs typically use 32K-100K tokens.*

**Industry choices**:
- GPT-2: 50,257 tokens (BPE)
- BERT: 30,522 tokens (WordPiece)
- LLaMA 2: 32,000 tokens (SentencePiece BPE)
- GPT-4/Claude: ~100K tokens (optimized for diverse inputs)

The trend is toward larger vocabularies as models scale—the fixed cost of more embeddings is offset by shorter sequences and better coverage.

#### Why Larger Vocabularies Pay Off at Scale

Think of it as **fixed cost vs. variable cost**:

**Embedding cost (fixed, one-time):**
```
Memory increase = (100K - 32K) × embedding_dim × 4 bytes
                = 68K × 4096 × 4 ≈ 1.1 GB
```
This is paid once, regardless of how much text you process.

**Sequence length cost (variable, every inference):**

Transformer attention is $O(n^2)$ where $n$ is sequence length.

| Vocabulary | "tokenization" becomes | Tokens |
|------------|----------------------|--------|
| Small (char-level) | `["t","o","k","e","n","i","z","a","t","i","o","n"]` | 12 |
| Large | `["token", "ization"]` | 2 |

Every inference pays this cost. Process a trillion tokens, and the savings compound massively.

**The trade-off:**

| Factor | Small Vocab (8K) | Large Vocab (100K) |
|--------|------------------|-------------------|
| Embedding memory | Small (one-time) | Large (one-time) |
| Sequence length | Long | **Short** |
| Per-inference compute | High ($n^2$) | **Low** |
| Rare word coverage | Poor | **Good** |

**Analogy**: It's like buying vs. renting a house:
- **Large vocab = Buying**: High down payment (big embedding table), but low monthly cost (short sequences)
- **Small vocab = Renting**: Low upfront cost, but you pay more every month (long sequences)

When the model is already tens of GB, an extra 1-2 GB for embeddings is negligible. But the per-inference savings accumulate forever.

---

## 4. Special Tokens: The Control Panel

Beyond regular text tokens, every tokenizer includes **special tokens** that serve as control signals:

![Special Tokens](../zh/images/day07/special-tokens.png)
*Figure 5: Special tokens provide essential signals for model training and inference. Different architectures use different special tokens.*

| Token | Full Name | Purpose |
|-------|-----------|---------|
| `[PAD]` | Padding | Fill sequences to uniform length for batching |
| `[UNK]` | Unknown | Fallback for out-of-vocabulary tokens (rare with subword) |
| `[BOS]` / `<s>` | Begin of Sequence | Signals start of generation |
| `[EOS]` / `</s>` | End of Sequence | Signals completion; model should stop |
| `[SEP]` | Separator | Separates segments (e.g., question from answer) |
| `[CLS]` | Classification | BERT's aggregate representation token |
| `[MASK]` | Mask | BERT's masked language modeling objective |

**Critical for prompting**: Many subtle prompt engineering issues stem from special token handling. For example, adding `[EOS]` at the wrong place can make a model stop prematurely.

Modern chat models add even more special tokens:
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
<|im_start|>assistant
```

These delimit roles and enable multi-turn conversation structure.

---

## 5. Popular Tokenizers: A Practical Comparison

Different organizations have made different choices:

![Tokenizer Comparison](../zh/images/day07/tokenizer-comparison.png)
*Figure 6: Major tokenizers used by different models. Tiktoken (OpenAI) is notable for its Rust implementation, making it extremely fast.*

### GPT Tokenizers (BPE)

OpenAI's tokenizers have evolved:
- **GPT-2**: 50,257 tokens, byte-level BPE
- **GPT-3.5/4**: `cl100k_base` with 100,277 tokens
- **GPT-4o**: Optimized for multilingual, ~200K tokens

"Byte-level" means the base vocabulary is 256 bytes, not characters. This handles any UTF-8 text without `[UNK]`.

### BERT Tokenizer (WordPiece)

BERT uses 30,522 tokens with WordPiece. The `##` prefix indicates word continuation:
```
"tokenization" → ["token", "##ization"]
"unhappiness"  → ["un", "##happiness"]  # or further split
```

### Tiktoken: Speed Matters

OpenAI open-sourced [tiktoken](https://github.com/openai/tiktoken), a blazingly fast BPE implementation in Rust with Python bindings. For production systems processing millions of requests, tokenization speed matters:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
print(enc.decode(tokens))  # "Hello, world!"
```

**Why "Fast Rust impl" matters**: Tiktoken is written in Rust (not pure Python), making tokenization 3-6x faster. This is critical when processing millions of API requests—every millisecond saved in tokenization adds up.

### Why Claude Uses ~100K Tokens (Larger Than Many)

You might notice Claude's vocabulary (~100K) is larger than BERT (30K) or LLaMA 2 (32K). Several reasons:

1. **Better multilingual support**: More tokens for Chinese, Japanese, Arabic, etc.
2. **Shorter sequences**: As discussed earlier, larger vocab = shorter sequences = faster inference. Critical for Claude's 200K context window.
3. **Better code coverage**: Programming keywords (`function`, `const`, `async`) as single tokens
4. **Industry trend**: GPT-4 also uses ~100K; LLaMA 3 jumped from 32K to 128K

The extra embedding memory is negligible for large models, but the inference speedup compounds over billions of requests.

---

## 6. Tokenization Edge Cases and Gotchas

Understanding tokenization quirks explains many puzzling LLM behaviors:

![Tokenization Edge Cases](../zh/images/day07/tokenization-edge-cases.png)
*Figure 7: Common edge cases that catch developers by surprise. These tokenization quirks directly impact model behavior.*

### Numbers Are Fragmented

The number "2024" might tokenize as `["20", "24"]` or even `["2", "0", "2", "4"]`. This fragmentation is why LLMs struggle with:
- Arithmetic ("What is 17 * 24?")
- Counting ("How many r's in strawberry?")
- Number comparison ("Is 12345 greater than 9999?")

The model never sees numbers as unified entities—just arbitrary symbol sequences.

### Whitespace Sensitivity

A leading space creates different tokens:
```python
enc.encode("hello")   # [15339]
enc.encode(" hello")  # [24748]  # Different token!
```

This matters for:
- Prompt formatting (template concatenation)
- Code completion (indentation)
- Reproducibility across implementations

### Multilingual Inefficiency

Tokenizers trained primarily on English text are inefficient for other languages:

| Text | English Tokens | Equivalent Meaning in Spanish |
|------|---------------|-------------------------------|
| "Hello, how are you?" | 5 tokens | "Hola, ¿cómo estás?" = 7+ tokens |

This means:
- Non-English users pay more per API call
- Context windows hold fewer non-English words
- Some languages face 2-5x token inflation

### Code Tokenization

Programming languages have their own quirks:
```python
def calculate_sum(a, b):
    return a + b
```
Might tokenize as many separate pieces, with indentation and symbols fragmented unnaturally. Specialized code models often use custom tokenizers.

---

## 7. Code Example: Exploring Tokenization

Let's experiment with different tokenizers:

```python
import tiktoken
from transformers import AutoTokenizer

# OpenAI's tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Test strings
texts = [
    "Hello, world!",
    "tokenization",
    "🚀 Let's go!",
    "def func(x): return x * 2",
    "2024年3月31日",  # Japanese/Chinese date
]

print("=" * 60)
print("OpenAI cl100k_base (GPT-4)")
print("=" * 60)
for text in texts:
    tokens = enc.encode(text)
    print(f"'{text}'")
    print(f"  Tokens: {tokens}")
    print(f"  Count: {len(tokens)}")
    print(f"  Decoded pieces: {[enc.decode([t]) for t in tokens]}")
    print()

# Compare with BERT tokenizer
print("=" * 60)
print("BERT WordPiece")
print("=" * 60)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

for text in texts[:3]:  # BERT struggles with emoji
    tokens = bert_tokenizer.tokenize(text.lower())
    ids = bert_tokenizer.encode(text, add_special_tokens=False)
    print(f"'{text}'")
    print(f"  Tokens: {tokens}")
    print(f"  IDs: {ids}")
    print(f"  Count: {len(tokens)}")
    print()
```

**Sample output** (simplified):
```
'Hello, world!'
  Tokens: [9906, 11, 1917, 0]
  Count: 4
  Decoded pieces: ['Hello', ',', ' world', '!']

'tokenization'
  Tokens: [5765, 2065]
  Count: 2
  Decoded pieces: ['token', 'ization']

'🚀 Let's go!'
  Tokens: [9468, 248, 10058, 596, 733, 0]
  Count: 6
  Decoded pieces: ['🚀', '', " Let's", ' go', '!']
```

**Key observations**:
1. Common words stay intact; rare words split
2. Emoji use multiple tokens (encoded as byte sequences)
3. Space handling varies (" world" vs "world")
4. Different tokenizers produce different splits

---

## 8. Math Derivation: Information Theory Perspective [Optional]

> This section is for readers who want deeper understanding.

Tokenization can be viewed through the lens of information theory. The goal is to find a vocabulary that minimizes the total description length of the corpus.

**Minimum Description Length (MDL) principle**:

$$
\begin{aligned}
L_{\text{total}} &= L_{\text{vocab}} + L_{\text{corpus}} \\
&= |\mathcal{V}| \cdot \text{avg\_token\_len} + \sum_{t \in \text{corpus}} \log_2 \frac{1}{P(t)}
\end{aligned}
$$

Where:
- $L_{\text{vocab}}$: Cost of storing the vocabulary
- $L_{\text{corpus}}$: Cost of encoding the corpus using the vocabulary
- $P(t)$: Probability of token $t$ in the corpus

**BPE approximates this**: By merging frequent pairs, BPE creates tokens that appear often, reducing $L_{\text{corpus}}$. The fixed vocabulary size limits $L_{\text{vocab}}$.

**Unigram (alternative to BPE)** explicitly optimizes:

$$
\mathcal{L}(\mathbf{x}) = \sum_{i=1}^{N} \log P(x_i | \theta)
$$

Where $\mathbf{x}$ is the corpus and $\theta$ represents vocabulary parameters. Unigram starts with a large vocabulary and iteratively prunes tokens that least impact the likelihood.

---

## 9. Common Misconceptions

### ❌ "Tokenization is just preprocessing—it doesn't really matter"

**Reality**: Tokenization fundamentally shapes what patterns a model can learn. A model that sees "2024" as `["20", "24"]` literally cannot learn that 2024 is a single number. Tokenization is a **design choice** with far-reaching consequences.

### ❌ "More tokens = longer text"

**Reality**: Token count doesn't correlate directly with word count. "Hello" is 1 token, but "🚀" might be 3 tokens (encoded as UTF-8 bytes). This is why API pricing by tokens can surprise users writing in non-English or using emoji.

### ❌ "All tokenizers work the same way"

**Reality**: GPT-4's tokenizer produces completely different token IDs than BERT's. You cannot mix tokenizers—a model trained with one tokenizer must use that exact tokenizer at inference.

---

## 10. Historical Context: From Characters to Subwords

The evolution of tokenization mirrors the field's growth:

| Era | Approach | Limitation |
|-----|----------|------------|
| 1990s | Word-level | OOV problem, massive vocabularies |
| 2000s | Character-level | Very long sequences, slow training |
| 2016 | BPE (Sennrich et al.) | First practical subword approach |
| 2018 | WordPiece (BERT) | Optimized for language understanding |
| 2018 | SentencePiece | Language-agnostic, no preprocessing |
| 2020+ | Byte-level BPE | Handles any UTF-8, no [UNK] |

The seminal 2016 paper by Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), showed that BPE dramatically improved translation quality for rare words. This insight spread to all of NLP.

---

## 11. Practical Implications

### For API Users

- **Count tokens, not words**: Use `tiktoken` or the provider's tokenizer to estimate costs
- **Non-English is expensive**: Expect 1.5-3x more tokens for non-English text
- **Code is fragmented**: Programming languages often tokenize inefficiently
- **Prompt engineering**: Be aware that whitespace and special characters affect tokenization

### For Model Developers

- **Vocabulary size**: Balance between sequence length and embedding memory
- **Multilingual support**: Consider SentencePiece or byte-level BPE for global applications
- **Domain-specific**: Financial, medical, or code models may benefit from custom tokenizers
- **Evaluation**: Token-level metrics (perplexity) don't always correlate with word-level quality

---

## 12. Further Reading

### Beginner
1. [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)
   Visualize how text is tokenized in real-time

2. [Hugging Face Tokenizers Tutorial](https://huggingface.co/docs/tokenizers/)
   Comprehensive guide to training custom tokenizers

### Advanced
1. [tiktoken GitHub Repository](https://github.com/openai/tiktoken)
   OpenAI's fast BPE implementation with detailed documentation

2. [SentencePiece GitHub](https://github.com/google/sentencepiece)
   Google's language-agnostic tokenizer library

### Papers
1. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
   Sennrich et al. (2016) — The BPE paper that started it all

2. [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)
   Kudo & Richardson (2018) — Introduces SentencePiece

3. [A Systematic Comparison of BPE and Byte-Level Models](https://arxiv.org/abs/2004.03720)
   Detailed analysis of tokenization strategies

---

## Reflection Questions

1. **Why do you think GPT models use larger vocabularies (100K+) while earlier models used smaller ones (30K)?** Consider the trade-offs between sequence length, memory, and coverage.

2. **How might tokenization affect a model's ability to learn mathematical reasoning?** Think about how numbers are represented and what patterns become visible or invisible.

3. **If you were building a specialized LLM for a non-English language, what tokenization choices would you make differently from English-centric models?**

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Tokenization | Converting text to numerical IDs that models can process |
| Subword tokenization | Balance between word-level and character-level approaches |
| BPE | Iteratively merge frequent character pairs to build vocabulary |
| WordPiece | Similar to BPE but maximizes training data likelihood |
| SentencePiece | Language-agnostic tokenizer treating input as raw bytes |
| Vocabulary size | Trade-off between sequence length and embedding memory |
| Special tokens | Control signals like [PAD], [EOS], [CLS] for model behavior |
| Token fragmentation | Why LLMs struggle with numbers, code, and non-English |

**Key Takeaway**: Tokenization is the invisible foundation of language models. It determines what patterns are learnable, how efficiently text is processed, and why models exhibit certain quirks. Understanding tokenization transforms mysterious LLM behaviors into predictable consequences of design choices. The next time GPT can't count letters in "strawberry," you'll know exactly why—and appreciate the elegant engineering that makes everything else work.

---

*Day 7 of 60 | LLM Fundamentals*
*Word count: ~3,200 | Reading time: ~16 minutes*
