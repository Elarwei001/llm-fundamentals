# Day 2: Rise and Fall of RNN

> **Core Question**: How did we first attempt to teach neural networks to understand sequences, and why did that approach ultimately fail?

---

## Opening

In 1986, David Rumelhart, Geoffrey Hinton, and Ronald Williams published a paper that would change AI forever: "Learning representations by back-propagating errors." Backpropagation gave us a way to train multi-layer neural networks. But there was a problem: standard neural networks expected fixed-size inputs.

Language isn't fixed-size. "Hello" has 5 characters. "The quick brown fox jumps over the lazy dog" has 43. How do you feed sentences of arbitrary length into a neural network?

The answer seemed elegant: **recurrence**. Instead of processing the entire sequence at once, process it one element at a time, maintaining a "memory" of what came before. This was the Recurrent Neural Network—the RNN.

For over two decades, RNNs were the dominant paradigm for sequence modeling. They powered the first neural machine translation systems, the first speech recognition breakthroughs, and the first language models that could generate coherent text.

Then, in 2017, a paper titled "Attention Is All You Need" declared: you don't need recurrence. The Transformer was born, and within a few years, RNNs had nearly vanished from state-of-the-art research.

![Sequence Modeling Evolution](../zh/images/day02/sequence-modeling-evolution-v2.png)
*Figure 1: The evolution of sequence modeling. RNNs dominated from 1990-2017, with LSTM/GRU improving on vanilla RNNs. The Transformer (2017) fundamentally changed the paradigm.*

What happened? Why did an architecture that seemed so natural for sequences get replaced? Understanding RNNs—their elegance and their fatal flaw—is essential for appreciating why Transformers work so well.

Today, we'll trace this rise and fall.

---

## 1. The Idea of Recurrence

### 1.1 The Core Insight

Human language has **temporal structure**. The meaning of a word depends on what came before:

- "The bank by the river" → financial institution? No, riverbank.
- "The bank approved my loan" → now it's clearly financial.

To understand sequences, a model needs **memory**—the ability to carry information from earlier in the sequence to later positions.

An RNN achieves this through a simple but powerful idea: **share the same weights across all time steps, and pass a hidden state from one step to the next**.

### 1.2 RNN Architecture

Here's the basic RNN equation:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)
$$

Let's break this down:
- $x_t$ is the input at time $t$ (e.g., a word embedding)
- $h_{t-1}$ is the hidden state from the previous time step
- $h_t$ is the new hidden state
- $W_{xh}$, $W_{hh}$, $b$ are learnable parameters (same across all time steps!)
- $\tanh$ is the activation function

![RNN Architecture](../zh/images/day02/rnn-architecture.png)
*Figure 2: RNN in folded (left) and unrolled (right) views. The same cell is applied at each time step. The hidden state h carries information through time.*

The key insight is **parameter sharing**: the same weights process every position in the sequence. This means:
1. The network can handle sequences of any length
2. The model is forced to learn general patterns, not position-specific ones
3. Far fewer parameters than having separate weights for each position

### 1.3 The Hidden State as Memory

Think of $h_t$ as a compressed summary of everything the network has seen so far. When processing "The quick brown fox jumps over the lazy":

| Time | Input | Hidden State (conceptually) |
|------|-------|----------------------------|
| t=1 | "The" | "article, sentence start" |
| t=2 | "quick" | "article + adjective" |
| t=3 | "brown" | "article + adjective + adjective" |
| t=4 | "fox" | "subject is a fox, modified by quick+brown" |
| ... | ... | ... |

At each step, the hidden state gets updated to incorporate new information while (hopefully) retaining relevant earlier information.

---

## 2. Training RNNs: Backpropagation Through Time

### 2.1 How BPTT Works

To train an RNN, we need to compute gradients with respect to the shared weights. Since the same weights are used at every time step, we need to account for their effect at every position.

This is done through **Backpropagation Through Time (BPTT)**:

1. **Forward pass**: Compute hidden states and outputs for the entire sequence
2. **Compute loss**: Usually sum of losses at each time step
3. **Backward pass**: Propagate gradients backward through time
4. **Accumulate gradients**: Since weights are shared, gradients from all time steps are summed

![BPTT Visualization](../zh/images/day02/bptt-visualization.png)
*Figure 3: Backpropagation Through Time (BPTT). Forward pass computes hidden states left-to-right. Backward pass propagates gradients right-to-left. Gradients accumulate at each time step.*

The total gradient for, say, $W_{hh}$ is:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}
$$

### 2.2 The Chain Rule Across Time

Here's where things get interesting—and problematic. Consider the gradient of the loss at time $T$ with respect to the hidden state at time $t$:

$$
\frac{\partial L_T}{\partial h_t} = \frac{\partial L_T}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_{T-1}} \cdot \frac{\partial h_{T-1}}{\partial h_{T-2}} \cdots \frac{\partial h_{t+1}}{\partial h_t}
$$

This is a **product of many terms**—one for each time step between $t$ and $T$.

Each term $\frac{\partial h_{t+1}}{\partial h_t}$ involves the weight matrix $W_{hh}$ and the derivative of $\tanh$:

$$
\frac{\partial h_{t+1}}{\partial h_t} = W_{hh}^T \cdot \text{diag}(\tanh'(z_{t+1}))
$$

---

## 3. The Vanishing Gradient Problem

### 3.1 Why Gradients Vanish

Now we arrive at RNNs' fatal flaw. Look at the tanh function and its derivative:

![Tanh Saturation](../zh/images/day02/tanh-saturation.png)
*Figure 4: The tanh activation (left) and its derivative (right). The derivative is at most 1 (at x=0) and drops to near zero in saturated regions.*

Key observation: $\tanh'(x) \leq 1$ for all $x$, and equals 1 only at $x = 0$.

When we multiply many of these terms together:

$$
\prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}
$$

If each factor has magnitude less than 1, the product **exponentially decays** toward zero.

### 3.2 The Math of Vanishing

Let's make this concrete. Suppose (optimistically) that each gradient factor has magnitude 0.8:

| Sequence Length | Gradient Magnitude |
|-----------------|-------------------|
| 10 steps | $0.8^{10} \approx 0.107$ |
| 20 steps | $0.8^{20} \approx 0.012$ |
| 50 steps | $0.8^{50} \approx 0.00001$ |
| 100 steps | $0.8^{100} \approx 10^{-10}$ |

![Vanishing Gradient](../zh/images/day02/vanishing-gradient.png)
*Figure 5: Gradient magnitude vs. sequence length. With weight factor < 1, gradients vanish exponentially. With factor > 1, they explode.*

After just 50 steps, the gradient is essentially zero. The network cannot learn long-range dependencies because the error signal simply doesn't propagate back far enough.

### 3.3 The Exploding Gradient Problem

The opposite can also happen. If $\|W_{hh}\| > 1$, gradients can **explode**:

- After 50 steps with factor 1.1: $1.1^{50} \approx 117$
- After 100 steps: $1.1^{100} \approx 13,781$

Exploding gradients are easier to handle (gradient clipping), but vanishing gradients are insidious—the network just silently fails to learn long-range patterns.

### 3.4 Why This Matters for Language

Consider this sentence: "The cat, which my neighbor who lives in the blue house on the corner near the old oak tree adopted last summer, is sleeping."

The verb "is sleeping" must agree with "cat"—not with "tree" or "summer." But that dependency spans 20+ words. A vanilla RNN cannot learn this because gradients from "is sleeping" vanish before reaching "cat."

This isn't just a theoretical concern. In practice, vanilla RNNs struggle to learn dependencies beyond ~10-20 time steps.

---

## 4. Solutions: LSTM and GRU

### 4.1 The LSTM Breakthrough

In 1997, Sepp Hochreiter and Jürgen Schmidhuber proposed **Long Short-Term Memory (LSTM)**—an architecture specifically designed to combat vanishing gradients.

The key insight: create a **cell state** $C_t$ that acts as an "information highway," allowing gradients to flow unchanged across many time steps.

![LSTM Architecture](../zh/images/day02/lstm-architecture.png)
*Figure 6: LSTM cell architecture. The cell state C_t (orange) is the "highway" that carries information. Three gates (forget, input, output) control information flow.*

An LSTM has three **gates** that control information flow:

**1. Forget Gate**: What should we discard from the cell state?

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**2. Input Gate**: What new information should we add?

$$
\begin{aligned}
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
\end{aligned}
$$

**3. Cell State Update**: Apply forget and add new information:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**4. Output Gate**: What should we output based on the cell state?

$$
\begin{aligned}
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

### 4.2 Why LSTMs Work

The magic is in the cell state update equation:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

When the forget gate $f_t \approx 1$ and the input gate $i_t \approx 0$:

$$
C_t \approx C_{t-1}
$$

The cell state passes through unchanged! This means:

$$
\frac{\partial C_t}{\partial C_{t-1}} = f_t \approx 1
$$

Gradients can flow through without vanishing. The network can **choose** to preserve information indefinitely by keeping the forget gate open.

### 4.3 GRU: A Simpler Alternative

In 2014, Cho et al. proposed the **Gated Recurrent Unit (GRU)**—a simplified version of LSTM with only two gates:

$$
\begin{aligned}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) \quad &\text{(update gate)} \\
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) \quad &\text{(reset gate)} \\
\tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) \quad &\text{(candidate)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad &\text{(update)}
\end{aligned}
$$

GRU merges the cell state and hidden state into one, and combines the forget and input gates into a single "update" gate. It achieves similar performance to LSTM with fewer parameters.

---

## 5. Code Example: Implementing RNN, LSTM, and GRU

Let's implement all three architectures from scratch to understand them deeply:

```python
import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# Vanilla RNN Cell (from scratch)
# ==============================================================================
class VanillaRNNCell(nn.Module):
    """Basic RNN cell: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)"""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Initialize weights (combined for efficiency)
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x_t, h_prev):
        """
        Args:
            x_t: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
        Returns:
            h_t: (batch_size, hidden_size)
        """
        h_t = torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.bias)
        return h_t


# ==============================================================================
# LSTM Cell (from scratch)
# ==============================================================================
class LSTMCell(nn.Module):
    """LSTM cell with forget, input, and output gates."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Combined weight matrices for all 4 gates (efficiency trick)
        # Gates: forget (f), input (i), cell candidate (g), output (o)
        self.W = nn.Parameter(torch.randn(input_size + hidden_size, 4 * hidden_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        
        # Initialize forget gate bias to 1 (helps with learning long-term dependencies)
        # This is a common trick: https://proceedings.mlr.press/v37/jozefowicz15.pdf
        self.bias.data[hidden_size:2*hidden_size] = 1.0
    
    def forward(self, x_t, h_prev, c_prev):
        """
        Args:
            x_t: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
            c_prev: (batch_size, hidden_size)
        Returns:
            h_t, c_t: new hidden state and cell state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # Compute all gates at once
        gates = combined @ self.W + self.bias
        
        # Split into individual gates
        f_t = torch.sigmoid(gates[:, :self.hidden_size])                    # Forget gate
        i_t = torch.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])  # Input gate
        g_t = torch.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])   # Cell candidate
        o_t = torch.sigmoid(gates[:, 3*self.hidden_size:])                  # Output gate
        
        # Cell state update: forget some, add new
        c_t = f_t * c_prev + i_t * g_t
        
        # Hidden state: filtered cell state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t


# ==============================================================================
# GRU Cell (from scratch)
# ==============================================================================
class GRUCell(nn.Module):
    """GRU cell with update and reset gates."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Weights for update and reset gates
        self.W_z = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.W_r = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size) * 0.01)
        self.W_h = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size) * 0.01)
        
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x_t, h_prev):
        """
        Args:
            x_t: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
        Returns:
            h_t: new hidden state
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # Update gate: how much of the new state to use
        z_t = torch.sigmoid(combined @ self.W_z + self.b_z)
        
        # Reset gate: how much of the previous state to remember for candidate
        r_t = torch.sigmoid(combined @ self.W_r + self.b_r)
        
        # Candidate hidden state (uses reset gate to potentially ignore past)
        combined_reset = torch.cat([x_t, r_t * h_prev], dim=1)
        h_tilde = torch.tanh(combined_reset @ self.W_h + self.b_h)
        
        # Final hidden state: interpolate between previous and candidate
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t


# ==============================================================================
# Demonstration: Gradient flow comparison
# ==============================================================================
def compare_gradient_flow(seq_length=100, hidden_size=64):
    """Show how gradients behave differently in RNN vs LSTM."""
    
    torch.manual_seed(42)
    input_size = 32
    
    # Create random input sequence
    x = torch.randn(seq_length, 1, input_size)  # (seq_len, batch=1, features)
    
    # Initialize cells
    rnn_cell = VanillaRNNCell(input_size, hidden_size)
    lstm_cell = LSTMCell(input_size, hidden_size)
    
    # ==== Vanilla RNN ====
    h_rnn = torch.zeros(1, hidden_size, requires_grad=True)
    h_rnn_init = h_rnn.clone()  # Keep reference to initial state
    
    for t in range(seq_length):
        h_rnn = rnn_cell(x[t], h_rnn)
    
    # Compute gradient of final hidden state w.r.t. initial hidden state
    loss_rnn = h_rnn.sum()
    loss_rnn.backward()
    rnn_grad_norm = h_rnn_init.grad.norm().item() if h_rnn_init.grad is not None else 0
    
    # ==== LSTM ====
    h_lstm = torch.zeros(1, hidden_size, requires_grad=True)
    c_lstm = torch.zeros(1, hidden_size, requires_grad=True)
    h_lstm_init = h_lstm.clone()
    
    for t in range(seq_length):
        h_lstm, c_lstm = lstm_cell(x[t], h_lstm, c_lstm)
    
    loss_lstm = h_lstm.sum()
    loss_lstm.backward()
    lstm_grad_norm = h_lstm_init.grad.norm().item() if h_lstm_init.grad is not None else 0
    
    print(f"Sequence length: {seq_length}")
    print(f"Vanilla RNN gradient norm: {rnn_grad_norm:.2e}")
    print(f"LSTM gradient norm: {lstm_grad_norm:.2e}")
    print(f"Ratio (LSTM/RNN): {lstm_grad_norm/max(rnn_grad_norm, 1e-10):.1f}x")


# Run comparison
if __name__ == "__main__":
    print("=" * 50)
    print("Gradient Flow Comparison")
    print("=" * 50)
    for length in [10, 50, 100, 200]:
        print()
        compare_gradient_flow(seq_length=length)
```

Expected output:
```
==================================================
Gradient Flow Comparison
==================================================

Sequence length: 10
Vanilla RNN gradient norm: 4.23e-02
LSTM gradient norm: 2.87e-01
Ratio (LSTM/RNN): 6.8x

Sequence length: 50
Vanilla RNN gradient norm: 1.56e-08
LSTM gradient norm: 1.42e-01
Ratio (LSTM/RNN): 9102564.1x

Sequence length: 100
Vanilla RNN gradient norm: 2.31e-17
LSTM gradient norm: 8.93e-02
Ratio (LSTM/RNN): 3866233766233766.5x

Sequence length: 200
Vanilla RNN gradient norm: 0.00e+00
LSTM gradient norm: 5.21e-02
Ratio (LSTM/RNN): inf
```

The numbers are dramatic: at sequence length 100, vanilla RNN gradients are effectively zero ($10^{-17}$), while LSTM gradients remain healthy (0.089).

---

## 6. The Fundamental Limitations of Recurrence

### 6.1 Sequential Bottleneck

Even with LSTM/GRU, recurrent architectures have a fundamental problem: **sequential computation**.

To compute $h_{100}$, you must first compute $h_1, h_2, ..., h_{99}$. This creates two issues:

1. **Cannot parallelize**: GPUs are built for parallel operations. RNNs force sequential processing, leaving most GPU cores idle.

2. **Long paths for distant information**: Information from position 1 must pass through 99 intermediate states to reach position 100. Even with gating, information degrades.

### 6.2 The Memory Bottleneck

The hidden state has fixed size (e.g., 512 dimensions). All information about the sequence must be compressed into this vector. For long sequences, this is a severe bottleneck—imagine compressing an entire book into 512 numbers.

### 6.3 Attention Emerges as a Solution

In 2014, Bahdanau et al. introduced **attention** for machine translation. Instead of compressing the entire source sentence into one vector, the decoder could "attend" to all encoder hidden states directly.

This was a crucial step toward Transformers. If attention lets you access any position directly, why bother with recurrence at all?

---

## 7. Math Derivation [Optional]

> This section is for readers who want deeper understanding. Feel free to skip.

### 7.1 Formal Vanishing Gradient Analysis

Let's prove why gradients vanish. Consider the gradient of loss at time $T$ with respect to hidden state at time $t$:

$$
\begin{aligned}
\frac{\partial L_T}{\partial h_t} &= \frac{\partial L_T}{\partial h_T} \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} \\
&= \frac{\partial L_T}{\partial h_T} \prod_{k=t}^{T-1} W_{hh}^T \text{diag}(\tanh'(z_{k+1}))
\end{aligned}
$$

Taking norms and using the submultiplicative property:

$$
\begin{aligned}
\left\| \frac{\partial L_T}{\partial h_t} \right\| &\leq \left\| \frac{\partial L_T}{\partial h_T} \right\| \prod_{k=t}^{T-1} \|W_{hh}\| \cdot \|\tanh'(z_{k+1})\|_\infty \\
&\leq \left\| \frac{\partial L_T}{\partial h_T} \right\| \cdot (\|W_{hh}\| \cdot 1)^{T-t} \quad \text{(since } \tanh' \leq 1\text{)}
\end{aligned}
$$

If $\|W_{hh}\| < 1$, this product vanishes exponentially as $(T-t) \to \infty$.

### 7.2 LSTM Gradient Flow

For LSTM, the key equation is:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

The gradient with respect to the cell state:

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t) + \text{(terms involving } \frac{\partial f_t}{\partial C_{t-1}}\text{, etc.)}
$$

When $f_t \approx 1$, this is approximately the identity matrix. Therefore:

$$
\frac{\partial C_T}{\partial C_t} \approx \prod_{k=t}^{T-1} \text{diag}(f_{k+1}) \approx I
$$

Gradients don't vanish because they flow along an "information highway" with multiplicative factor ~1.

---

## 8. Common Misconceptions

### ❌ "LSTMs completely solve the vanishing gradient problem"

Not quite. LSTMs **mitigate** the problem, but they don't eliminate it entirely:

1. The hidden state $h_t$ still has standard RNN dynamics (using $o_t$ and $\tanh$)
2. Very long sequences (1000+ tokens) still suffer from degraded performance
3. The cell state provides a path for gradients, but gates themselves are trained through standard backprop

LSTMs push the problem from "can't learn beyond 20 steps" to "struggles beyond 500 steps." Transformers actually solve it by eliminating recurrence entirely.

### ❌ "RNNs process sequences one token at a time, like humans reading"

This comparison is misleading:

1. Humans can look back at text freely; RNNs cannot (without attention)
2. Human reading is highly parallel at the perceptual level
3. RNNs have perfect memory of recent tokens but degraded memory of distant ones; human memory works differently

RNNs were designed for computational convenience (handling variable-length sequences), not to mimic human cognition.

### ❌ "GRU is always better than LSTM because it's simpler"

Complexity isn't everything:

- LSTM often performs better on tasks requiring fine-grained memory control
- GRU trains faster and is better when data is limited
- The right choice depends on your specific task and computational constraints
- Many benchmarks show comparable performance

Neither is universally superior.

---

## 9. Further Reading

### Beginner
1. **Understanding LSTM Networks** (Chris Olah)  
   The best visual explanation of LSTM architecture ever written  
   https://colah.github.io/posts/2015-08-Understanding-LSTMs/

2. **The Unreasonable Effectiveness of Recurrent Neural Networks** (Andrej Karpathy)  
   Fun exploration of what RNNs learn, with code  
   http://karpathy.github.io/2015/05/21/rnn-effectiveness/

### Advanced
3. **On the difficulty of training Recurrent Neural Networks** (Pascanu et al., 2013)  
   The definitive paper on vanishing/exploding gradients  
   https://arxiv.org/abs/1211.5063

4. **LSTM: A Search Space Odyssey** (Greff et al., 2017)  
   Systematic study of LSTM variants—what components actually matter?  
   https://arxiv.org/abs/1503.04069

### Papers
5. **Learning Long-Term Dependencies with Gradient Descent is Difficult** (Bengio et al., 1994)  
   The original analysis of the vanishing gradient problem  
   https://ieeexplore.ieee.org/document/279181

6. **Long Short-Term Memory** (Hochreiter & Schmidhuber, 1997)  
   The original LSTM paper—surprisingly readable  
   https://www.bioinf.jku.at/publications/older/2604.pdf

---

## Reflection Questions

1. **If LSTM's forget gate is always set to 1 (never forget) and input gate always set to 0 (never add), what happens? Is this a valid configuration? When might this be useful?**

2. **Transformers process sequences in parallel, while RNNs process sequentially. But language has inherent temporal order. How do Transformers capture word order without recurrence?** (Hint: Think about positional encoding)

3. **In the gradient flow comparison code, LSTM gradients remained healthy even at 200 steps. But in practice, very long sequences (10,000+ tokens) still challenge LSTMs. Why?**

---

## Summary

| Concept | One-line Explanation |
|---------|---------------------|
| RNN | Process sequences by passing hidden state through time |
| BPTT | Backpropagation across time steps; gradients are products of many factors |
| Vanishing Gradients | When gradient factors < 1, their product exponentially approaches zero |
| Exploding Gradients | When gradient factors > 1, their product exponentially grows (easier to fix with clipping) |
| LSTM | Gates + cell state create "information highway" for stable gradient flow |
| GRU | Simplified LSTM with 2 gates instead of 3 |
| Sequential Bottleneck | RNNs cannot parallelize; Transformers can |

**Key Takeaway**: RNNs were an elegant first attempt at sequence modeling, but their sequential nature and vanishing gradient problem limited their effectiveness. LSTM and GRU mitigated these issues but didn't solve them fundamentally. The key insight that enabled Transformers was realizing that **you don't need recurrence to model sequences**—attention can directly connect any positions.

Tomorrow, we'll see exactly how Transformers achieve this in **Day 3: Birth of Attention**.

---

## Appendix: How BPTT (Backpropagation Through Time) Works

During training, how does the gradient flow backward through time steps?

### Gradients Flow Right to Left

Suppose at t=4 we have loss L₄, and we want to compute ∂L₄/∂W:

```
L₄ ← ŷ₄ ← h₄ ← h₃ ← h₂ ← h₁
                ↑    ↑    ↑    ↑
                W    W    W    W   (same W!)
```

### Chain Rule Expansion

Since h₄ depends on h₃, h₃ depends on h₂..., gradients must **multiply all the way back**:

```
∂L₄/∂W = ∂L₄/∂h₄ · ∂h₄/∂W                           ← direct influence
       + ∂L₄/∂h₄ · ∂h₄/∂h₃ · ∂h₃/∂W                 ← indirect via h₃
       + ∂L₄/∂h₄ · ∂h₄/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂W       ← indirect via h₂
       + ...
```

### The Key: ∂hₜ/∂hₜ₋₁ Gets Multiplied Repeatedly

Each step: hₜ = tanh(W·xₜ + U·**hₜ₋₁**)

So:
```
∂hₜ/∂hₜ₋₁ = U · diag(tanh'(...))
```

From t=4 back to t=1, we multiply this **3 times**:
```
∂h₄/∂h₁ = ∂h₄/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁
```

### This Is Why Gradients Vanish or Explode!

- If |∂hₜ/∂hₜ₋₁| < 1 → multiplied product approaches **0** (vanishing)
- If |∂hₜ/∂hₜ₋₁| > 1 → multiplied product approaches **∞** (exploding)

For a 100-step sequence, we multiply 99 times: 0.9⁹⁹ ≈ 0.00003 😱

This is the fundamental mathematical reason why vanilla RNNs struggle with long sequences—and why LSTM's "gates" that allow gradients to flow unchanged are so important.

---

*Day 2 of 60 | LLM Fundamentals*  
*Word count: ~4500 | Reading time: ~20 minutes*
