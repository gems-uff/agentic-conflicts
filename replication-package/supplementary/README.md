# Supplementary Material

This directory contains supplementary analyses and stratification results not included in the main paper due to space constraints.

## Contents

### `by_language.md`

Conflict resolution patterns stratified by programming language.

**Includes:**
- Table: Conflict incidence by language (top 10)
- Table: Resolution strategy distribution by language
- Per-agent self-resolution rates by language
- Patterns and insights across different programming languages

### `by_task_type.md`

Conflict resolution patterns stratified by PR task type (feature addition, bug fix, refactoring, etc.).

**Includes:**
- Table: Conflict incidence by PR task type
- Table: Resolution strategy distribution by task type
- Per-agent self-resolution rates by task type
- Narrative on how agent behavior varies with the kind of work being performed

## Generating Supplementary Results

These analyses are automatically generated when running:

```bash
python launch_pipeline.py --analyze-only --data-dir ./data
```

Output files will be written to `../data/results/supplementary/`.
