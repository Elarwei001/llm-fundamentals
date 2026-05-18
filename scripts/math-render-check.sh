#!/bin/bash
# math-render-check.sh — Scan course articles for LaTeX rendering risks on GitHub
# Usage: bash math-render-check.sh [day_number]
#   If day_number is given, checks that day's zh+en articles.
#   If omitted, checks ALL articles.
#
# Detects:
#   1. Inline math ($...$) inside HTML tags (<table>, <td>, <div>, etc.)
#   2. Inline math ($...$) inside blockquotes (lines starting with >)
#   3. Inline math ($...$) inside figure/image captions (*...*)
#   4. Display math ($$...$$) inside HTML tags
#   5. Display math ($$...$$) inside blockquotes
#
# Exit codes:
#   0 = all clean (or only warnings)
#   1 = critical issues found (things that WILL break rendering)

set -euo pipefail

DAY="${1:-}"
ISSUES=0
WARNINGS=0

if [[ -n "$DAY" ]]; then
    DAY=$(printf "%02d" $((10#$DAY)))
    FILES=""
    for lang in zh en; do
        f=$(ls articles/${lang}/day${DAY}-*.md 2>/dev/null | head -1)
        if [[ -n "$f" ]]; then
            FILES="$FILES $f"
        fi
    done
else
    FILES=$(find articles/zh articles/en -name 'day*-*.md' 2>/dev/null | sort)
fi

if [[ -z "$FILES" ]]; then
    echo "No article files found."
    exit 1
fi

echo "=== Math Render Check ==="
echo ""

for FILE in $FILES; do
    FILE_ISSUES=0
    FILE_WARNINGS=0
    IN_HTML=false
    IN_DISPLAY=false
    DISPLAY_START=0
    LINE_NUM=0

    while IFS= read -r LINE; do
        LINE_NUM=$((LINE_NUM + 1))

        # Track HTML block context
        if echo "$LINE" | grep -qE '<(table|tr|td|th|div|span|details|summary)'; then
            IN_HTML=true
        fi
        if echo "$LINE" | grep -qE '</(table|tr|td|th|div|span|details|summary)>'; then
            # Stay in HTML until closing tag of outermost block
            if echo "$LINE" | grep -qE '</table>'; then
                IN_HTML=false
            fi
        fi

        # Track display math context
        if echo "$LINE" | grep -qE '^\s*\$\$\s*$'; then
            if $IN_DISPLAY; then
                IN_DISPLAY=false
            else
                IN_DISPLAY=true
                DISPLAY_START=$LINE_NUM
            fi
            continue
        fi

        # Skip content inside display math (it usually renders fine)
        if $IN_DISPLAY; then
            continue
        fi

        # --- CHECK 1: Inline math in blockquote ---
        if echo "$LINE" | grep -qE '^\s*>' && echo "$LINE" | grep -qE '\$[^$]+\$'; then
            echo "⚠️  $FILE:$LINE_NUM — inline math in blockquote (may not render)"
            echo "   $LINE" | head -c 120
            echo ""
            FILE_WARNINGS=$((FILE_WARNINGS + 1))
        fi

        # --- CHECK 2: Inline math in HTML context ---
        if $IN_HTML && echo "$LINE" | grep -qE '\$[^$]+\$'; then
            echo "❌ $FILE:$LINE_NUM — inline math inside HTML tags (WILL NOT render)"
            echo "   $LINE" | head -c 120
            echo ""
            FILE_ISSUES=$((FILE_ISSUES + 1))
        fi

        # --- CHECK 3: Display math in HTML context ---
        if $IN_HTML && echo "$LINE" | grep -qE '^\s*\$\$'; then
            echo "❌ $FILE:$LINE_NUM — display math inside HTML tags (WILL NOT render)"
            echo "   $LINE" | head -c 120
            echo ""
            FILE_ISSUES=$((FILE_ISSUES + 1))
        fi

        # --- CHECK 4: Display math in blockquote ---
        if echo "$LINE" | grep -qE '^\s*>\s*\$\$'; then
            echo "❌ $FILE:$LINE_NUM — display math in blockquote (WILL NOT render)"
            echo "   $LINE" | head -c 120
            echo ""
            FILE_ISSUES=$((FILE_ISSUES + 1))
        fi

        # --- CHECK 5: Inline math in figure caption (*...*) ---
        # Captions are lines starting with * and ending with *
        STRIPPED=$(echo "$LINE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if echo "$STRIPPED" | grep -qE '^\*.*\*$' && echo "$STRIPPED" | grep -qE '\$[^$]+\$'; then
            echo "⚠️  $FILE:$LINE_NUM — inline math in image caption (may not render)"
            echo "   $LINE" | head -c 120
            echo ""
            FILE_WARNINGS=$((FILE_WARNINGS + 1))
        fi

    done < "$FILE"

    # Unclosed display math
    if $IN_DISPLAY; then
        echo "❌ $FILE:$DISPLAY_START — unclosed display math block (\$\$ without closing)"
        FILE_ISSUES=$((FILE_ISSUES + 1))
    fi

    if [[ $FILE_ISSUES -gt 0 || $FILE_WARNINGS -gt 0 ]]; then
        echo "   → $FILE: $FILE_ISSUES critical, $FILE_WARNINGS warnings"
        echo ""
    else
        echo "✅ $FILE — clean"
    fi

    ISSUES=$((ISSUES + FILE_ISSUES))
    WARNINGS=$((WARNINGS + FILE_WARNINGS))
done

echo ""
echo "=== Summary: $ISSUES critical issues, $WARNINGS warnings ==="

if [[ $ISSUES -gt 0 ]]; then
    echo "❌ FAIL — critical rendering issues found. Fix before pushing."
    exit 1
else
    if [[ $WARNINGS -gt 0 ]]; then
        echo "⚠️  PASS with warnings — review recommended."
    else
        echo "✅ ALL CLEAN"
    fi
    exit 0
fi
