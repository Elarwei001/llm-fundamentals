#!/usr/bin/env python3
"""Generate figures for Day 7: Tokenization"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 150

# Color palette
COLORS = {
    'blue': '#3498db',
    'green': '#2ecc71',
    'orange': '#e67e22',
    'red': '#e74c3c',
    'purple': '#9b59b6',
    'gray': '#95a5a6',
    'dark': '#2c3e50',
    'light_blue': '#85c1e9',
    'light_green': '#82e0aa',
    'light_orange': '#f5b041',
}

def save_fig(fig, name):
    fig.savefig(f'/tmp/personal-skills/llm-fundamentals/articles/zh/images/day07/{name}.png', 
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Saved {name}.png')


# ============================================================
# Figure 1: Why Tokenization Matters - The Pipeline
# ============================================================
def create_pipeline_figure():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('From Text to Model: Why Tokenization is the Bridge', fontsize=16, fontweight='bold', pad=20)
    
    # Input text
    ax.add_patch(patches.FancyBboxPatch((0.3, 1.3), 2.5, 1.4, 
                                         boxstyle="round,pad=0.1", 
                                         facecolor=COLORS['light_blue'], edgecolor=COLORS['blue'], linewidth=2))
    ax.text(1.55, 2.4, 'Raw Text', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(1.55, 1.8, '"Hello world"', ha='center', va='center', fontsize=10, style='italic')
    
    # Arrow 1
    ax.annotate('', xy=(3.2, 2), xytext=(2.9, 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Tokenizer box
    ax.add_patch(patches.FancyBboxPatch((3.4, 1.0), 2.8, 2.0, 
                                         boxstyle="round,pad=0.1", 
                                         facecolor=COLORS['light_orange'], edgecolor=COLORS['orange'], linewidth=2))
    ax.text(4.8, 2.5, 'Tokenizer', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(4.8, 1.9, 'BPE / WordPiece', ha='center', va='center', fontsize=10)
    ax.text(4.8, 1.4, 'SentencePiece', ha='center', va='center', fontsize=10)
    
    # Arrow 2
    ax.annotate('', xy=(6.6, 2), xytext=(6.3, 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Token IDs
    ax.add_patch(patches.FancyBboxPatch((6.8, 1.3), 2.5, 1.4, 
                                         boxstyle="round,pad=0.1", 
                                         facecolor=COLORS['light_green'], edgecolor=COLORS['green'], linewidth=2))
    ax.text(8.05, 2.4, 'Token IDs', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(8.05, 1.8, '[15496, 995]', ha='center', va='center', fontsize=10, family='monospace')
    
    # Arrow 3
    ax.annotate('', xy=(9.7, 2), xytext=(9.4, 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Embedding Layer
    ax.add_patch(patches.FancyBboxPatch((9.9, 1.0), 2.2, 2.0, 
                                         boxstyle="round,pad=0.1", 
                                         facecolor='#d7bde2', edgecolor=COLORS['purple'], linewidth=2))
    ax.text(11, 2.5, 'Embedding', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(11, 1.9, 'Lookup Table', ha='center', va='center', fontsize=10)
    ax.text(11, 1.4, '(Vocab × Dim)', ha='center', va='center', fontsize=9)
    
    # Arrow 4
    ax.annotate('', xy=(12.5, 2), xytext=(12.2, 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Vectors
    ax.add_patch(patches.FancyBboxPatch((12.7, 1.3), 1.2, 1.4, 
                                         boxstyle="round,pad=0.1", 
                                         facecolor='#fadbd8', edgecolor=COLORS['red'], linewidth=2))
    ax.text(13.3, 2.4, 'Vectors', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(13.3, 1.8, '[0.2, -0.1,', ha='center', va='center', fontsize=8, family='monospace')
    ax.text(13.3, 1.5, ' 0.8, ...]', ha='center', va='center', fontsize=8, family='monospace')
    
    save_fig(fig, 'tokenization-pipeline')


# ============================================================
# Figure 2: Character vs Word vs Subword
# ============================================================
def create_comparison_figure():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Three Tokenization Strategies', fontsize=16, fontweight='bold', y=1.02)
    
    # Example text
    text = "unhappiness"
    
    # Character-level
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Character-level', fontsize=14, fontweight='bold', color=COLORS['blue'])
    
    chars = list(text)
    for i, c in enumerate(chars):
        x = (i % 4) * 2.2 + 1
        y = 4.5 - (i // 4) * 1.5
        ax.add_patch(patches.FancyBboxPatch((x-0.4, y-0.5), 0.8, 1.0,
                                             boxstyle="round,pad=0.05",
                                             facecolor=COLORS['light_blue'], edgecolor=COLORS['blue']))
        ax.text(x, y, c, ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(5, 0.8, f'Tokens: {len(chars)}', ha='center', fontsize=11, color=COLORS['dark'])
    ax.text(5, 0.2, 'Vocab size: ~100', ha='center', fontsize=10, color=COLORS['gray'])
    
    # Word-level
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Word-level', fontsize=14, fontweight='bold', color=COLORS['green'])
    
    ax.add_patch(patches.FancyBboxPatch((2.5, 2.5), 5, 1.2,
                                         boxstyle="round,pad=0.1",
                                         facecolor=COLORS['light_green'], edgecolor=COLORS['green']))
    ax.text(5, 3.1, text, ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(5, 0.8, 'Tokens: 1', ha='center', fontsize=11, color=COLORS['dark'])
    ax.text(5, 0.2, 'Vocab size: ~500K+', ha='center', fontsize=10, color=COLORS['gray'])
    
    # Subword-level (BPE)
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Subword (BPE)', fontsize=14, fontweight='bold', color=COLORS['orange'])
    
    subwords = ['un', 'happi', 'ness']
    colors_sub = [COLORS['light_orange'], '#f9e79f', '#abebc6']
    x_positions = [1.5, 4.5, 7.5]
    widths = [1.5, 2.2, 2.0]
    
    for i, (sw, col, x, w) in enumerate(zip(subwords, colors_sub, x_positions, widths)):
        ax.add_patch(patches.FancyBboxPatch((x-w/2, 2.5), w, 1.2,
                                             boxstyle="round,pad=0.1",
                                             facecolor=col, edgecolor=COLORS['orange']))
        ax.text(x, 3.1, sw, ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(5, 0.8, 'Tokens: 3', ha='center', fontsize=11, color=COLORS['dark'])
    ax.text(5, 0.2, 'Vocab size: ~30K-100K', ha='center', fontsize=10, color=COLORS['gray'])
    
    plt.tight_layout()
    save_fig(fig, 'tokenization-strategies')


# ============================================================
# Figure 3: BPE Algorithm Steps
# ============================================================
def create_bpe_algorithm_figure():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('BPE Algorithm: Learning Merge Rules', fontsize=16, fontweight='bold', pad=20)
    
    # Step boxes
    steps = [
        ('Step 0: Initialize', 'Start with characters', ['l o w </w>', 'l o w e r </w>', 'n e w e s t </w>'], COLORS['light_blue']),
        ('Step 1: Count pairs', 'Most frequent: (e, s) = 2', ['l o w </w>', 'l o w e r </w>', 'n e w es t </w>'], COLORS['light_green']),
        ('Step 2: Merge (e,s)', 'Most frequent: (es, t) = 2', ['l o w </w>', 'l o w e r </w>', 'n e w est </w>'], COLORS['light_orange']),
        ('Step 3: Merge (es,t)', 'Most frequent: (l, o) = 2', ['l o w </w>', 'l o w e r </w>', 'n e w est </w>'], '#d7bde2'),
    ]
    
    y_start = 9
    for i, (title, note, tokens, color) in enumerate(steps):
        y = y_start - i * 2.3
        
        # Title
        ax.text(0.2, y, title, fontsize=12, fontweight='bold', color=COLORS['dark'])
        ax.text(4, y, note, fontsize=10, style='italic', color=COLORS['gray'])
        
        # Token boxes
        for j, token in enumerate(tokens):
            ax.add_patch(patches.FancyBboxPatch((0.2 + j * 3.8, y - 1.4), 3.5, 0.9,
                                                 boxstyle="round,pad=0.05",
                                                 facecolor=color, edgecolor=COLORS['dark'], alpha=0.8))
            ax.text(0.2 + j * 3.8 + 1.75, y - 0.95, token, ha='center', va='center', 
                    fontsize=10, family='monospace')
        
        # Arrow between steps
        if i < len(steps) - 1:
            ax.annotate('', xy=(6, y - 1.7), xytext=(6, y - 2.1),
                        arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    
    # Final vocabulary box
    ax.add_patch(patches.FancyBboxPatch((0.2, 0.2), 11.5, 1.3,
                                         boxstyle="round,pad=0.1",
                                         facecolor='#fdebd0', edgecolor=COLORS['orange'], linewidth=2))
    ax.text(6, 1.2, 'Final Vocabulary (after many iterations)', ha='center', fontsize=12, fontweight='bold')
    ax.text(6, 0.6, 'Base chars + Learned merges: {a, b, ..., z, </w>, lo, low, est, er, new, ...}', 
            ha='center', fontsize=10, family='monospace')
    
    save_fig(fig, 'bpe-algorithm')


# ============================================================
# Figure 4: Vocabulary Size Trade-off
# ============================================================
def create_vocab_tradeoff_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Vocab size vs sequence length
    ax = axes[0]
    vocab_sizes = [100, 1000, 10000, 32000, 100000, 500000]
    seq_lengths = [100, 25, 8, 4, 2.5, 1.2]  # relative to word count
    
    ax.plot(vocab_sizes, seq_lengths, 'o-', color=COLORS['blue'], linewidth=2, markersize=8)
    ax.fill_between(vocab_sizes, seq_lengths, alpha=0.2, color=COLORS['blue'])
    ax.set_xscale('log')
    ax.set_xlabel('Vocabulary Size', fontsize=12)
    ax.set_ylabel('Avg Tokens per Word', fontsize=12)
    ax.set_title('Larger Vocab → Shorter Sequences', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 1000000)
    
    # Annotations
    ax.annotate('Character\n(too long)', (100, 100), fontsize=9, ha='center',
                xytext=(100, 70), arrowprops=dict(arrowstyle='->', color=COLORS['gray']))
    ax.annotate('GPT-2/3\n(sweet spot)', (50000, 3), fontsize=9, ha='center',
                xytext=(150000, 15), arrowprops=dict(arrowstyle='->', color=COLORS['green']))
    
    # Right: Memory vs OOV rate
    ax = axes[1]
    
    vocab_sizes_2 = np.array([1000, 5000, 10000, 32000, 100000, 500000])
    memory = vocab_sizes_2 * 768 * 4 / 1e6  # Assuming 768-dim embeddings, 4 bytes per float
    oov_rate = np.array([15, 5, 2, 0.5, 0.1, 0.01])  # approximate OOV rates
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(vocab_sizes_2, memory, 'o-', color=COLORS['red'], linewidth=2, markersize=8, label='Memory (MB)')
    ax.fill_between(vocab_sizes_2, memory, alpha=0.15, color=COLORS['red'])
    ax.set_ylabel('Embedding Memory (MB)', color=COLORS['red'], fontsize=12)
    ax.tick_params(axis='y', labelcolor=COLORS['red'])
    
    line2 = ax2.plot(vocab_sizes_2, oov_rate, 's--', color=COLORS['green'], linewidth=2, markersize=8, label='OOV Rate (%)')
    ax2.set_ylabel('OOV Rate (%)', color=COLORS['green'], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLORS['green'])
    ax2.set_yscale('log')
    
    ax.set_xscale('log')
    ax.set_xlabel('Vocabulary Size', fontsize=12)
    ax.set_title('Memory vs Coverage Trade-off', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=10)
    
    plt.tight_layout()
    save_fig(fig, 'vocab-size-tradeoff')


# ============================================================
# Figure 5: Special Tokens
# ============================================================
def create_special_tokens_figure():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Special Tokens: The Control Panel of LLMs', fontsize=16, fontweight='bold', pad=15)
    
    tokens = [
        ('[PAD]', 'Padding', 'Fill sequences to\nsame length', COLORS['gray'], 0.5),
        ('[UNK]', 'Unknown', 'Fallback for\nunseen tokens', COLORS['red'], 2.5),
        ('[BOS]', 'Begin of Seq', 'Start generation\nhere', COLORS['green'], 4.5),
        ('[EOS]', 'End of Seq', 'Stop generation\nhere', COLORS['blue'], 6.5),
        ('[SEP]', 'Separator', 'Separate segments\n(BERT)', COLORS['purple'], 8.5),
        ('[CLS]', 'Classification', 'Aggregate repr\n(BERT)', COLORS['orange'], 10.5),
    ]
    
    for token, name, desc, color, x in tokens:
        # Token box
        ax.add_patch(patches.FancyBboxPatch((x - 0.8, 4.5), 1.6, 1.2,
                                             boxstyle="round,pad=0.1",
                                             facecolor=color, edgecolor='white', linewidth=2, alpha=0.85))
        ax.text(x, 5.1, token, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Name
        ax.text(x, 3.8, name, ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['dark'])
        
        # Description
        ax.text(x, 2.8, desc, ha='center', va='center', fontsize=9, color=COLORS['gray'])
    
    # Example sentence at bottom
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 11, 1.5,
                                         boxstyle="round,pad=0.1",
                                         facecolor='#fef9e7', edgecolor=COLORS['orange'], linewidth=1.5))
    ax.text(6, 1.6, 'Example (BERT Input):', ha='center', fontsize=10, fontweight='bold')
    ax.text(6, 1.0, '[CLS] How are you ? [SEP] I am fine . [SEP] [PAD] [PAD]', 
            ha='center', fontsize=10, family='monospace', color=COLORS['dark'])
    
    save_fig(fig, 'special-tokens')


# ============================================================
# Figure 6: Tokenizer Comparison Table
# ============================================================
def create_tokenizer_comparison_figure():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_title('Popular Tokenizers Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Table data
    headers = ['Tokenizer', 'Algorithm', 'Used By', 'Vocab Size', 'Key Feature']
    rows = [
        ['GPT-2/3/4', 'BPE', 'OpenAI', '50,257', 'Byte-level BPE'],
        ['BERT', 'WordPiece', 'Google', '30,522', 'Subword with ##'],
        ['T5/Gemma', 'SentencePiece', 'Google', '32,000', 'Language-agnostic'],
        ['LLaMA', 'BPE (SP)', 'Meta', '32,000', 'Byte-fallback'],
        ['Claude', 'BPE variant', 'Anthropic', '~100K', 'Unknown details'],
        ['Tiktoken', 'BPE', 'OpenAI', '100,277', 'Fast Rust impl'],
    ]
    
    # Create table
    cell_colors = [['#d5f5e3'] * 5]  # header color
    for i in range(len(rows)):
        cell_colors.append(['#fef9e7' if i % 2 == 0 else '#ffffff'] * 5)
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors[1:],
        colColours=['#d5f5e3'] * 5,
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_text_props(fontweight='bold')
    
    save_fig(fig, 'tokenizer-comparison')


# ============================================================
# Figure 7: Tokenization Edge Cases
# ============================================================
def create_edge_cases_figure():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Tokenization Edge Cases & Gotchas', fontsize=16, fontweight='bold', y=1.02)
    
    cases = [
        ('Numbers', '"2024" → ["20", "24"]', 'GPT struggles with\narithmetic partly\ndue to tokenization', COLORS['blue']),
        ('Whitespace', '" hello" ≠ "hello"', 'Leading space creates\ndifferent tokens\n(context matters!)', COLORS['green']),
        ('Multilingual', '"Bonjour" → 2+ tokens', 'Non-English often\ngets more tokens\n(higher cost)', COLORS['orange']),
        ('Code', '"def func():" → 4+ tokens', 'Indentation and\nsymbols fragment\nunnaturally', COLORS['purple']),
    ]
    
    for ax, (title, example, desc, color) in zip(axes.flatten(), cases):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Title
        ax.add_patch(patches.FancyBboxPatch((0.5, 4.5), 9, 1.2,
                                             boxstyle="round,pad=0.1",
                                             facecolor=color, edgecolor='white', linewidth=2, alpha=0.85))
        ax.text(5, 5.1, title, ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        # Example
        ax.text(5, 3.5, example, ha='center', va='center', fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.9))
        
        # Description
        ax.text(5, 1.5, desc, ha='center', va='center', fontsize=10, color=COLORS['dark'])
    
    plt.tight_layout()
    save_fig(fig, 'tokenization-edge-cases')


# ============================================================
# Generate all figures
# ============================================================
if __name__ == '__main__':
    print('Generating Day 7 figures...')
    create_pipeline_figure()
    create_comparison_figure()
    create_bpe_algorithm_figure()
    create_vocab_tradeoff_figure()
    create_special_tokens_figure()
    create_tokenizer_comparison_figure()
    create_edge_cases_figure()
    print('All figures generated!')
