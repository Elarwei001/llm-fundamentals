#!/bin/bash
# pre-check.sh - 运行质量门禁检查
# Usage: bash pre-check.sh 05  (for Day 05)

DAY=$1

if [[ -z "$DAY" ]]; then
    echo "Usage: bash pre-check.sh <day_number>"
    echo "Example: bash pre-check.sh 05"
    exit 1
fi

# Pad day number with zero if needed
DAY=$(printf "%02d" $((10#$DAY)))

ARTICLE=$(ls articles/en/day${DAY}-*.md 2>/dev/null | head -1)
IMAGES="articles/zh/images/day${DAY}/"

echo "=== Pre-check for Day ${DAY} ==="
echo ""

# 1. 文章文件存在
if [[ ! -f "$ARTICLE" ]]; then
    echo "❌ FAIL: Article file not found for day ${DAY}"
    exit 1
fi
echo "✅ Article exists: $ARTICLE"

# 2. 图片文件存在 (>= 3)
IMG_COUNT=$(ls ${IMAGES}/*.png 2>/dev/null | wc -l | tr -d ' ')
if [[ $IMG_COUNT -lt 3 ]]; then
    echo "❌ FAIL: Only $IMG_COUNT images (need >= 3)"
    exit 1
fi
echo "✅ Images exist: $IMG_COUNT files"

# 3. 图片嵌入文章 (>= 3 个 ![...] 引用)
EMBED_COUNT=$(grep -c '!\[' "$ARTICLE" 2>/dev/null || echo "0")
if [[ $EMBED_COUNT -lt 3 ]]; then
    echo "❌ FAIL: Only $EMBED_COUNT image embeds in article (need >= 3)"
    exit 1
fi
echo "✅ Images embedded: $EMBED_COUNT references"

# 4. 有数学公式 (>= 1)
MATH_COUNT=$(grep -cE '\$[^$]+\$|\$\$|\\\[|\\\(' "$ARTICLE" 2>/dev/null || echo "0")
if [[ $MATH_COUNT -lt 1 ]]; then
    echo "❌ FAIL: No math formulas found"
    exit 1
fi
echo "✅ Math formulas: found"

# 5. 字数足够 (>= 1500 words)
WORD_COUNT=$(wc -w < "$ARTICLE" | tr -d ' ')
if [[ $WORD_COUNT -lt 1500 ]]; then
    echo "❌ FAIL: Only $WORD_COUNT words (need >= 1500)"
    exit 1
fi
echo "✅ Word count: $WORD_COUNT"

# 6. 无占位符链接
PLACEHOLDER=$(grep -c '\[Link\]' "$ARTICLE" 2>/dev/null || echo "0")
if [[ $PLACEHOLDER -gt 0 ]]; then
    echo "❌ FAIL: $PLACEHOLDER placeholder [Link] found"
    exit 1
fi
echo "✅ No placeholder links"

# 7. 有类比解释 (检查常见类比词汇) - WARNING only
ANALOGY=$(grep -ciE 'like a|similar to|think of|imagine|analogy|just like|比如|类似|想象|好比' "$ARTICLE" 2>/dev/null || echo "0")
if [[ $ANALOGY -lt 1 ]]; then
    echo "⚠️  WARNING: No analogies detected (recommended but not blocking)"
else
    echo "✅ Analogies: found"
fi

echo ""
echo "=== PRE-CHECK PASSED ==="
exit 0
