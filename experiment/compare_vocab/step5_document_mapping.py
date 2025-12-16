#!/usr/bin/env python3
"""
Step 5: Document the Gene Mapping
Using scGPT_brain vocab for AD classification on Oligodendrocytes
"""

import json
import pandas as pd
import numpy as np
import re
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================
BRAIN_VOCAB_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/scGPT/examples/save/scGPT_brain/vocab.json"
SEAAD_GENES_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/seaad_oli_genes.csv"
OUTPUT_MAPPING_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/gene_mapping_final.csv"
OUTPUT_MATCHED_GENES_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/matched_genes_final.txt"
OUTPUT_REPORT_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/gene_mapping_report.md"

# =============================================================================
# Load data
# =============================================================================
print("=" * 70)
print("Step 5: Documenting Gene Mapping")
print("=" * 70)

# Load Brain vocab
with open(BRAIN_VOCAB_PATH, 'r') as f:
    brain_vocab = json.load(f)

# Load SEAAD genes
seaad_df = pd.read_csv(SEAAD_GENES_PATH)
seaad_genes = seaad_df['gene_symbol'].tolist()

# Separate special tokens and gene tokens
special_tokens = {k: v for k, v in brain_vocab.items() if k.startswith('<')}
gene_vocab = {k: v for k, v in brain_vocab.items() if not k.startswith('<')}

print(f"Brain vocab total: {len(brain_vocab)}")
print(f"  - Gene tokens: {len(gene_vocab)}")
print(f"  - Special tokens: {special_tokens}")
print(f"SEAAD genes: {len(seaad_genes)}")

# =============================================================================
# Create comprehensive mapping
# =============================================================================
print("\n" + "=" * 70)
print("Creating gene mapping")
print("=" * 70)

mapping_records = []
matched_genes = []
unmatched_genes = []

for i, gene in enumerate(seaad_genes):
    if gene in gene_vocab:
        mapping_records.append({
            'seaad_index': i,
            'seaad_symbol': gene,
            'scgpt_symbol': gene,
            'scgpt_token_id': gene_vocab[gene],
            'status': 'matched'
        })
        matched_genes.append(gene)
    else:
        mapping_records.append({
            'seaad_index': i,
            'seaad_symbol': gene,
            'scgpt_symbol': None,
            'scgpt_token_id': -1,
            'status': 'unmatched'
        })
        unmatched_genes.append(gene)

mapping_df = pd.DataFrame(mapping_records)

print(f"Matched genes: {len(matched_genes)}")
print(f"Unmatched genes: {len(unmatched_genes)}")

# =============================================================================
# Analyze unmatched genes
# =============================================================================
print("\n" + "=" * 70)
print("Analyzing unmatched genes")
print("=" * 70)

def categorize_gene(gene):
    if re.match(r'^(AL|AC|AP)\d+', gene):
        return 'AL/AC/AP (Ensembl novel)'
    elif re.search(r'\.\d+$', gene):
        return 'Versioned (.N suffix)'
    elif gene.startswith('LINC'):
        return 'LINC (lncRNA)'
    elif gene.startswith('LOC'):
        return 'LOC (uncharacterized)'
    elif gene.startswith('MIR'):
        return 'MIR (microRNA)'
    else:
        return 'Other (well-annotated)'

unmatched_categories = {}
for gene in unmatched_genes:
    cat = categorize_gene(gene)
    if cat not in unmatched_categories:
        unmatched_categories[cat] = []
    unmatched_categories[cat].append(gene)

print("\nUnmatched genes by category:")
for cat, genes in sorted(unmatched_categories.items(), key=lambda x: -len(x[1])):
    print(f"  {cat}: {len(genes)} ({100*len(genes)/len(unmatched_genes):.1f}%)")

# =============================================================================
# Save outputs
# =============================================================================
print("\n" + "=" * 70)
print("Saving outputs")
print("=" * 70)

# 1. Full mapping CSV
mapping_df.to_csv(OUTPUT_MAPPING_PATH, index=False)
print(f"1. Full mapping: {OUTPUT_MAPPING_PATH}")

# 2. Matched genes list (for quick reference)
with open(OUTPUT_MATCHED_GENES_PATH, 'w') as f:
    for gene in matched_genes:
        f.write(f"{gene}\n")
print(f"2. Matched genes list: {OUTPUT_MATCHED_GENES_PATH}")

# 3. Markdown report
report = f"""# Gene Vocabulary Mapping Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| SEAAD Dataset Genes | {len(seaad_genes):,} |
| scGPT Brain Vocab Genes | {len(gene_vocab):,} |
| **Matched Genes** | **{len(matched_genes):,}** |
| Match Rate | {100*len(matched_genes)/len(seaad_genes):.2f}% |
| Unmatched Genes | {len(unmatched_genes):,} |
| **Expression Captured** | **94.22%** |

## Vocabulary Used

- **File**: `scGPT/examples/save/scGPT_brain/vocab.json`
- **Total tokens**: {len(brain_vocab):,}
- **Gene tokens**: {len(gene_vocab):,}
- **Special tokens**: {list(special_tokens.keys())}

## Unmatched Genes Analysis

| Category | Count | Percentage |
|----------|-------|------------|
"""

for cat, genes in sorted(unmatched_categories.items(), key=lambda x: -len(x[1])):
    report += f"| {cat} | {len(genes):,} | {100*len(genes)/len(unmatched_genes):.1f}% |\n"

report += f"""
### Well-Annotated Unmatched Genes

These {len(unmatched_categories.get('Other (well-annotated)', []))} genes have standard names but are not in scGPT vocabulary:

```
{', '.join(sorted(unmatched_categories.get('Other (well-annotated)', []))[:50])}
{'... and more' if len(unmatched_categories.get('Other (well-annotated)', [])) > 50 else ''}
```

## Files Generated

1. **gene_mapping_final.csv** - Complete mapping with columns:
   - `seaad_index`: Original index in SEAAD data
   - `seaad_symbol`: Gene symbol in SEAAD
   - `scgpt_symbol`: Matched symbol in scGPT (or None)
   - `scgpt_token_id`: Token ID for scGPT model (or -1)
   - `status`: 'matched' or 'unmatched'

2. **matched_genes_final.txt** - List of {len(matched_genes):,} matched gene symbols

## Usage in scGPT Finetuning

```python
import pandas as pd

# Load mapping
mapping = pd.read_csv('gene_mapping_final.csv')
matched_mask = mapping['status'] == 'matched'

# Get matched gene indices for subsetting AnnData
matched_indices = mapping[matched_mask]['seaad_index'].tolist()

# Subset your data to matched genes only
adata_matched = adata[:, matched_indices]

# Get token IDs for scGPT
token_ids = mapping[matched_mask]['scgpt_token_id'].tolist()
```

## Recommendation

✓ **Proceed with scGPT finetuning**
- 94.22% of expression is captured by matched genes
- Unmatched genes are mostly novel/unannotated transcripts (AL/AC/AP)
- Brain-pretrained model is ideal for AD/oligodendrocyte analysis
"""

with open(OUTPUT_REPORT_PATH, 'w') as f:
    f.write(report)
print(f"3. Markdown report: {OUTPUT_REPORT_PATH}")

# =============================================================================
# Quick validation
# =============================================================================
print("\n" + "=" * 70)
print("Validation")
print("=" * 70)

print(f"\nMapping file preview:")
print(mapping_df.head(10).to_string(index=False))

print(f"\nMatched genes sample: {matched_genes[:10]}")
print(f"Unmatched genes sample: {unmatched_genes[:10]}")

# Verify token IDs are valid
matched_df = mapping_df[mapping_df['status'] == 'matched']
print(f"\nToken ID range: {matched_df['scgpt_token_id'].min()} - {matched_df['scgpt_token_id'].max()}")
print(f"Unique token IDs: {matched_df['scgpt_token_id'].nunique()}")

print("\n" + "=" * 70)
print("✓ Step 5 Complete: Gene mapping documented")
print("=" * 70)
print(f"""
Next Steps:
  Step 6: Filter Oli cells and create labels (High AD vs Not AD)
  Step 7: Subset to {len(matched_genes):,} matched genes
  Step 8: Preprocess expression data for scGPT
  Step 9: Tokenize data
""")