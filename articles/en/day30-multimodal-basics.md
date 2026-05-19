# Day 30: Multimodal Basics — CLIP, Vision Transformer, GPT-4V

> **Core Question**: How do we teach a language model to see?

---

## Opening

You've spent 29 days learning how language models work. They read text, predict the next token, and somehow reason about the world. But here's the thing: the world isn't made of text. It's made of images, sounds, videos, and physical sensations.

If you showed GPT-3 a photo of a cat, it had no idea what you were talking about. It could write a sonnet about cats — but it couldn't *see* one.

That changed between 2021 and 2024. A series of breakthroughs — CLIP, Vision Transformer (ViT), and the bridging architecture behind GPT-4V — taught language models to see. Today, frontier models like Gemini 2.5 Pro and GPT-4.1 don't just process text; they natively understand images, video, and audio through a single transformer.

#### Intuition: The Translator Problem

Imagine you have a brilliant friend who only speaks Mandarin, and another who only speaks French. They're both experts, but they can't talk to each other. What do you need? A **translator** — someone who can convert ideas from one language into the other.

Multimodal AI faces the same problem. Vision encoders "speak" in spatial grids and pixel patterns. Language models "speak" in token embeddings. The entire history of multimodal AI is the story of building better translators between these two languages.

---

## 1. The Vision Transformer (ViT) — Images as Sentences

### 1.1 Why Not Use CNNs?

Convolutional Neural Networks (CNNs) dominated computer vision for a decade. They slide filters across images, building up features from edges to objects. But CNNs have a structural problem: they assume **locality**. Every filter only sees a small patch at a time.

The Transformer's superpower is **global attention** — every token can attend to every other token. The question was: can we treat an image like a sentence?

### 1.2 The ViT Insight: Patch Embedding

The Vision Transformer (proposed by Dosovitskiy et al. at Google Brain in 2020, paper: ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929)) answered this with a simple idea:

1. Split the image into fixed-size patches (e.g., 16×16 pixels)
2. Flatten each patch into a vector
3. Linearly project each vector into an embedding — just like a word token
4. Add a special `[CLS]` token and positional embeddings
5. Feed the whole sequence into a standard Transformer encoder

![Vision Transformer: From Image to Patches to Tokens](../zh/images/day30/vit-patch-embedding.png)
*Figure 1: The ViT pipeline — an image is split into patches, each projected into an embedding, and processed by a Transformer as if they were words in a sentence.*

#### Intuition: The Jigsaw Puzzle

Think of an image as a jigsaw puzzle. ViT breaks the puzzle into pieces (patches), assigns each piece a number (position embedding), and then lets the Transformer figure out how the pieces relate. The self-attention mechanism acts like someone studying how puzzle pieces connect — not just edge by edge, but all at once.

### 1.3 ViT vs CNN: When Scale Matters

| Aspect | CNN | ViT |
|--------|-----|-----|
| Inductive bias | Strong (locality, translation equivariance) | Weak (learned from data) |
| Data efficiency | Works well on small datasets | Needs large data (14M–300M images) |
| Global reasoning | Only in deep layers, via gradual receptive field | From layer 1, via self-attention |
| Compute scaling | Diminishing returns | Strong scaling with data and compute |

ViT didn't immediately replace CNNs. On small datasets like ImageNet (1.3M images), ViT underperformed ResNets. But when trained on massive datasets (JFT-300M, with 300M images), ViT pulled ahead — and scaled further than CNNs ever could.

This is the same scaling law story from Day 9: when you have enough data, architectures with weaker inductive bias but more flexibility tend to win.

---

## 2. CLIP — Learning to See by Reading

### 2.1 The Core Idea

CLIP (Contrastive Language-Image Pre-training), introduced by OpenAI in 2021 (paper: ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020)), asked a radical question: what if we don't train vision with labels like "cat" and "dog", but instead train it to understand natural language descriptions?

The training setup is elegant:

1. Collect 400 million (image, text) pairs from the internet
2. Encode each image through a ViT (or ResNet) vision encoder
3. Encode each text caption through a text Transformer
4. **Contrastive objective**: pull matching pairs together, push non-matching pairs apart

![How CLIP Learns: Contrastive Training](../zh/images/day30/clip-training-process.png)
*Figure 2: CLIP's contrastive training — matching image-text pairs are pulled close in embedding space, while non-matching pairs are pushed apart.*

#### Intuition: The Party Trick

Imagine a party where everyone wears a name tag. You've never seen these people, but you have to match each name tag to the right person. Your strategy: look at each person, read each name tag, and figure out which pairs make sense. "Tall person with basketball" matches "Michael." "Short person with paint" matches "Picasso."

CLIP does exactly this at massive scale. After training on 400M pairs, it learns that the embedding for an image of a cat should be close to the text "a photo of a cat" and far from "a photo of a car."

### 2.2 Zero-Shot Classification

CLIP's killer feature is **zero-shot classification**. Instead of training a classifier for specific categories, you just:

```
image = encode_image(photo)
text_cat = encode_text("a photo of a cat")
text_dog = encode_text("a photo of a dog")

# Which text is closest to the image?
similarity = cosine_similarity(image, [text_cat, text_dog])
prediction = argmax(similarity)
```

No fine-tuning needed. This matched ResNet-50 accuracy on ImageNet without ever training on ImageNet labels.

### 2.3 The CLIP Legacy

CLIP became the default vision encoder for nearly everything:
- **Stable Diffusion** uses CLIP's text encoder to guide image generation
- **LLaVA** uses CLIP's vision encoder to feed images into an LLM
- **DALL-E 2/3** uses CLIP to align text and image representations
- **SigLIP** (Google, 2023, ["Sigmoid Loss for Language Image Pre-Training"](https://arxiv.org/abs/2303.15343)) improved CLIP by replacing the softmax-based contrastive loss with a simpler sigmoid loss, removing the need for global normalization — making training more efficient and scalable

**SigLIP 2** (Google, February 2025, ["Multilingual Vision-Language Encoders with Improved Semantic Understanding"](https://arxiv.org/abs/2502.14786)) further advanced this line with self-supervised learning objectives and online data curation, delivering major gains in localization, dense prediction, and multilingual retrieval.

---

## 3. The Bridging Architecture — LLaVA and GPT-4V

### 3.1 The Problem: Vision and Language Speak Different Languages

By 2023, we had great vision encoders (CLIP/ViT) and great language models (LLaMA, GPT). But they lived in separate worlds. The vision encoder outputs spatial feature vectors; the LLM expects token embeddings.

How do you bridge them?

### 3.2 LLaVA: The Simple Bridge

LLaVA (Large Language-and-Vision Assistant), proposed by Liu et al. in 2023 (paper: ["Visual Instruction Tuning"](https://arxiv.org/abs/2304.08485)), introduced the simplest possible bridge:

1. Take a frozen CLIP ViT vision encoder
2. Take a frozen LLaMA language model
3. Add a **trainable projection layer** between them
4. Train only the projection layer on visual instruction data

The projection layer acts as a translator: it takes CLIP's visual features and converts them into the LLM's token embedding space. Once projected, the LLM treats image features just like text tokens.

#### Intuition: The Foreign Exchange Student

Think of it like a foreign exchange student. The student (vision encoder) has rich knowledge but speaks a different language. The school (LLM) has incredible reasoning ability but can't understand the student. The projection layer is the language class that teaches the student to express their knowledge in the school's language.

### 3.3 GPT-4V: The Proprietary Bridge

GPT-4V (GPT-4 with Vision), released by OpenAI in late 2023, likely uses a similar bridging architecture internally, but at massive scale. The key differences from LLaVA:

- Much larger vision encoder and LLM
- Likely trained on vastly more image-text pairs
- Additional training for spatial reasoning, OCR, and document understanding
- Enhanced safety guardrails for image inputs

OpenAI hasn't published the GPT-4V architecture details, but the general consensus in the research community is that it follows the "vision encoder → projection → LLM" paradigm, possibly with multiple cross-attention layers rather than a simple linear projection.

---

## 4. Three Eras of Multimodal Architecture

The field has evolved through three distinct architectural paradigms:

![Three Eras of Multimodal Architecture](../zh/images/day30/multimodal-evolution-eras.png)
*Figure 3: The evolution from dual encoders (CLIP), to bridging architectures (LLaVA, GPT-4V), to native multimodal transformers (Gemini, GPT-4o).*

### Era 1: Dual Encoders (2021)

Separate encoders for vision and text, connected only through a shared embedding space. Great for retrieval and zero-shot classification, but can't generate text about images.

- **CLIP** (OpenAI, 2021)
- **ALIGN** (Google, 2021, ["Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision"](https://arxiv.org/abs/2102.05918))

### Era 2: Bridging (2023)

Frozen vision encoder + projection layer + LLM. Simple, effective, and enabled the first wave of practical multimodal assistants.

- **LLaVA** (2023) — open-source, showed the approach works
- **GPT-4V** (2023) — proprietary, showed the approach scales
- **Qwen-VL** (Alibaba, 2023, ["Qwen-VL: A Versatile Vision-Language Model"](https://arxiv.org/abs/2308.12966)) — strong open-source alternative

### Era 3: Native Multimodal (2024–2026)

Single transformer trained from scratch on interleaved multimodal data. No separate encoders, no projection layer — the model processes text, image, audio, and video tokens through the same attention layers.

- **Gemini 1.0/2.5** (Google DeepMind, 2023–2025) — natively multimodal from the start
- **GPT-4o** (OpenAI, 2024) — "omni" model handling text, image, audio
- **LLaMA 4** (Meta, 2025, [Meta AI Blog](https://ai.meta.com/blog/llama-4-multimodal-ai/)) — integrates multimodal inputs in a unified transformer framework

#### Why Native Multimodal Wins

The bridging approach has a fundamental limitation: the vision encoder was trained independently, so its representations may not be optimal for the LLM's reasoning. Native multimodal models train everything jointly, allowing the vision and language representations to co-evolve. This leads to:
- Better visual reasoning (the model learns to "think visually")
- More efficient inference (no separate encoder + projection overhead)
- Seamless interleaving of modalities (e.g., "what changed between image 1 and image 2?")

---

## 5. How Cross-Attention Fuses Modalities

The technical mechanism that makes multimodal fusion work is **cross-attention**.

### 5.1 The Mechanism

In standard self-attention, queries (Q), keys (K), and values (V) all come from the same input. In cross-attention:

- **Q** comes from one modality (e.g., text tokens)
- **K, V** come from another modality (e.g., image tokens)

This lets text tokens "look up" relevant visual information, like a detective asking the photo lab for evidence related to a specific question.

![How Multimodal Models Fuse Vision and Language](../zh/images/day30/cross-attention-fusion.png)
*Figure 4: Cross-attention fusion — text tokens query image tokens to incorporate visual information into the language model's reasoning.*

### 5.2 Where It Happens

Different architectures place cross-attention at different depths:

| Architecture | Fusion Strategy | Cross-Attention Location |
|-------------|----------------|------------------------|
| LLaVA | Linear projection only | None (project once, then self-attention) |
| Flamingo (DeepMind, 2022, ["Flamingo: a Visual Language Model for Few-Shot Learning"](https://arxiv.org/abs/2204.14198)) | Cross-attention inserts | Every N-th layer in the LLM |
| Gemini | Native joint attention | Every layer (single model) |

The trend is clear: earlier architectures added cross-attention as an afterthought; modern architectures bake it into every layer.

### 5.3 The Math Behind CLIP's Contrastive Loss

CLIP's contrastive objective is the InfoNCE loss. For a batch of N image-text pairs:

$$
\begin{aligned}
L &= -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}
\end{aligned}
$$

Where sim(I, T) is the cosine similarity between image and text embeddings, and $\tau$ is a learned temperature parameter. This is essentially a softmax classifier where each image must find its matching text among N candidates (and vice versa).

---

## 6. Code Example: Using CLIP for Zero-Shot Classification

```python
import torch
from transformers import CLIPModel, CLIPProcessor

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare an image and candidate labels
from PIL import Image
image = Image.open("photo.jpg")

# Zero-shot classification: describe what the image might contain
candidate_texts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car",
    "a photo of a building"
]

# Encode both image and text
inputs = processor(text=candidate_texts, images=image, 
                   return_tensors="pt", padding=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    
# Compute similarity (cosine similarity between image and text embeddings)
logits = outputs.logits_per_image  # shape: [1, num_texts]
probs = logits.softmax(dim=1)

# Print results
for text, prob in zip(candidate_texts, probs[0]):
    print(f"{text}: {prob.item():.3f}")
# Output:
# a photo of a cat: 0.872
# a photo of a dog: 0.089
# a photo of a car: 0.023
# a photo of a building: 0.016
```

This works without any training on these specific categories. CLIP learned the visual concepts from 400M internet image-text pairs.

---

## 7. Common Misconceptions

### ❌ "Multimodal models just glue a vision model to a language model"

Early bridging architectures did approximately this, but even they required careful design of the projection layer and training data. Modern native multimodal models like Gemini are trained as single models from scratch — there's no "gluing" at all. The model learns joint representations across modalities.

### ❌ "CLIP understands images the way humans do"

CLIP learns statistical correspondences between images and text. It can match "a photo of a cat" to cat images with high accuracy, but it doesn't have human-like understanding of object relationships, physics, or spatial reasoning. A model might correctly label an image as "cat on table" while having no model of gravity or support surfaces.

### ❌ "ViT completely replaced CNNs"

ViT outperforms CNNs at scale, but CNNs remain competitive in data-scarce and compute-limited settings. Many production systems still use CNN-based backbones (especially MobileNet variants) for edge deployment. Hybrid architectures combining convolutional and attention layers are also common.

---

## 8. Frontier: What's New (2025–2026)

The multimodal landscape is moving fast:

1. **SigLIP 2** (Google, February 2025) — Upgraded vision-language encoders with self-supervised learning, delivering major gains in localization, dense prediction, and multilingual retrieval. Used as the default vision encoder in many 2025-era VLMs. ([arXiv](https://arxiv.org/abs/2502.14786))

2. **LLaVA-OneVision-1.5** (December 2025) — The latest from the LLaVA family adds reinforcement learning post-training to multimodal models. The 8B model outperforms Qwen2.5-VL-7B on 18 of 27 benchmarks, showing that RL-based alignment techniques transfer to the visual domain. ([arXiv](https://arxiv.org/abs/2509.23661), [GitHub](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5))

3. **Gemini 2.5 Pro** (Google, March 2025 onward) — Native multimodal with 1M token context window, processing text, images, audio, and video through a single model. Consistently tops multimodal benchmarks in 2026. ([Google AI Blog](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/))

4. **Native Multimodal Learning with Next-Token Prediction** (Nature, January 2026) — Research showing that next-token prediction, the same objective behind LLMs, can be extended to learn unified representations across text, images, and video in a single model. Validates the "one model, many modalities" paradigm. ([Nature](https://www.nature.com/articles/s41586-025-10041-x))

5. **LLaMA 4** (Meta, 2025) — Integrates multimodal inputs seamlessly in a unified transformer framework, jointly attending to textual and visual tokens. Represents Meta's entry into the native multimodal era. ([Meta AI Blog](https://ai.meta.com/blog/llama-4-multimodal-ai/))

6. **Efficient Edge VLMs** (Nature Communications, July 2025) — Lightweight multimodal LLMs achieving GPT-4V-level performance on edge devices, making multimodal AI accessible beyond cloud deployments. ([Nature Communications](https://www.nature.com/articles/s41467-025-61040-5))

---

## 9. Further Reading

### Foundational Papers

1. ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) — The original ViT paper (Dosovitskiy et al., 2020)
2. ["Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020) — The CLIP paper (Radford et al., 2021)
3. ["Visual Instruction Tuning"](https://arxiv.org/abs/2304.08485) — LLaVA (Liu et al., 2023)

### Key Architecture Papers

4. ["Flamingo: a Visual Language Model for Few-Shot Learning"](https://arxiv.org/abs/2204.14198) — Cross-attention based fusion (Alayrac et al., 2022)
5. ["Sigmoid Loss for Language Image Pre-Training"](https://arxiv.org/abs/2303.15343) — SigLIP (Zhai et al., 2023)
6. ["Qwen-VL: A Versatile Vision-Language Model"](https://arxiv.org/abs/2308.12966) — Strong open-source VLM (Bai et al., 2023)

### Recent Advances

7. ["SigLIP 2: Multilingual Vision-Language Encoders"](https://arxiv.org/abs/2502.14786) — Next-gen vision encoder (Google, February 2025)
8. ["Multimodal learning with next-token prediction"](https://www.nature.com/articles/s41586-025-10041-x) — Unified multimodal objective (Nature, January 2026)

### Deep Dive: The `[CLS]` Token

In ViT (and BERT), a special `[CLS]` token is prepended to the sequence:

```
[CLS]  patch_1  patch_2  patch_3  ...  patch_N
```

It **doesn't correspond to any patch** in the image — it's a learned embedding that starts from scratch. After multiple layers of self-attention, the `[CLS]` token interacts with all patches and **aggregates global information** about the entire image, which is then used for classification and other global tasks.

Intuitively, think of it as a meeting: all patch tokens are department heads giving reports, and `[CLS]` is the CEO. The CEO doesn't represent any department, but listens to all of them and delivers a final synthesized judgment.

#### Why is `[CLS]` better than GAP?

If you just need a whole-image representation, why not simply average all patch tokens (GAP, Global Average Pooling)?

- **GAP is a fixed operation** — all patches are treated equally. If the image has many background noise patches, they all contribute equally to the final representation.
- **`[CLS]` learns how to aggregate through attention** — it can learn to ignore irrelevant patches and focus on important ones, essentially performing a learnable weighted aggregation.

#### What happens if you don't use `[CLS]`?

It works fine! Alternatives include:

- **GAP (Global Average Pooling)**: Simply average all patch vectors. Simple but can't distinguish patch importance. Swin Transformer (Microsoft, 2021) uses GAP instead of `[CLS]`.
- **Attention Pooling**: Use a learned query vector to attend over patches, letting the model decide how much weight each patch should get. This is essentially a variant of `[CLS]`.

In practice, many modern models no longer use `[CLS]`. The original ViT paper adopted BERT's design largely for **historical inertia** — the authors wanted to prove that "treating images as sentences with NLP architecture works for vision," so keeping the architecture as close to BERT as possible made the argument more convincing. Strictly speaking, `[CLS]` isn't optimal, but it's not a bad choice either.

---

## Reflection Questions

1. Why does ViT need more data than CNNs to reach the same accuracy? What does this tell us about the role of inductive bias in neural architectures?

2. If CLIP can already do zero-shot classification, why do we need more complex architectures like LLaVA or Gemini? What can those do that CLIP cannot?

3. As we move toward native multimodal models, does the distinction between "vision model" and "language model" disappear? What are the implications for how we think about AI capabilities?

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| Vision Transformer (ViT) | Treats image patches like word tokens, using self-attention for global visual reasoning |
| CLIP | Trains vision and text encoders jointly via contrastive learning on 400M image-text pairs |
| Contrastive Learning | Pull matching pairs together, push non-matching pairs apart in embedding space |
| Zero-shot Classification | Classify images into arbitrary categories without specific training data |
| Bridging Architecture | Projection layer converts vision features into LLM token embeddings |
| Cross-Attention | Text tokens query image features to incorporate visual information |
| Native Multimodal | Single transformer trained on interleaved multimodal data from scratch |
| SigLIP / SigLIP 2 | Improved CLIP variants with sigmoid loss and self-supervised objectives |

**Key Takeaway**: The journey from CLIP to Gemini represents a fundamental shift in how AI processes information — from separate, specialized models connected by adapters, to single models that natively understand multiple modalities. The same Transformer architecture that revolutionized language is now becoming a universal information processing engine, treating images, audio, and video as just another kind of "language."

---

*Day 30 of 60 | LLM Fundamentals*
*Word count: ~2400 | Reading time: ~12 minutes*
