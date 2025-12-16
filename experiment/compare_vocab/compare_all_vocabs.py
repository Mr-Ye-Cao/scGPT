#!/usr/bin/env python3
"""
Compare all three vocabularies:
1. default_gene_vocab.json (Original)
2. default_census_vocab.json (Census)
3. scGPT_brain/vocab.json (Brain-specific)
"""

import json
import pandas as pd
import numpy as np
import re

# =============================================================================
# PATHS
# =============================================================================
ORIGINAL_VOCAB_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/scGPT/scgpt/tokenizer/default_gene_vocab.json"
CENSUS_VOCAB_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/scGPT/scgpt/tokenizer/default_census_vocab.json"
BRAIN_VOCAB_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/scGPT/examples/save/scGPT_brain/vocab.json"
SEAAD_GENES_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/seaad_oli_genes.csv"
OUTPUT_MAPPING_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/gene_mapping_brain_vocab.csv"

# =============================================================================
# Load all vocabularies
# =============================================================================
print("=" * 70)
print("Step 1: Loading all vocabularies")
print("=" * 70)

with open(ORIGINAL_VOCAB_PATH, 'r') as f:
    original_vocab = json.load(f)

with open(CENSUS_VOCAB_PATH, 'r') as f:
    census_vocab = json.load(f)

with open(BRAIN_VOCAB_PATH, 'r') as f:
    brain_vocab = json.load(f)

# Load SEAAD genes
seaad_df = pd.read_csv(SEAAD_GENES_PATH)
seaad_genes = seaad_df['gene_symbol'].tolist()

print(f"Original vocab size: {len(original_vocab):,}")
print(f"Census vocab size:   {len(census_vocab):,}")
print(f"Brain vocab size:    {len(brain_vocab):,}")
print(f"SEAAD genes:         {len(seaad_genes):,}")

# Get gene sets (exclude special tokens)
original_genes = set(k for k in original_vocab.keys() if not k.startswith('<'))
census_genes = set(k for k in census_vocab.keys() if not k.startswith('<'))
brain_genes = set(k for k in brain_vocab.keys() if not k.startswith('<'))
seaad_set = set(seaad_genes)

# Check for special tokens
print(f"\nSpecial tokens in Brain vocab: {[k for k in brain_vocab.keys() if k.startswith('<')]}")

# =============================================================================
# Match SEAAD genes against each vocab
# =============================================================================
print("\n" + "=" * 70)
print("Step 2: Matching SEAAD genes against each vocabulary")
print("=" * 70)

original_matched = seaad_set & original_genes
census_matched = seaad_set & census_genes
brain_matched = seaad_set & brain_genes

print(f"""
MATCHING RESULTS:
─────────────────────────────────────────────────────────────────────
                        Vocab Size    SEAAD Matched    Match Rate
─────────────────────────────────────────────────────────────────────
Original vocab:         {len(original_genes):,}        {len(original_matched):,}           {100*len(original_matched)/len(seaad_genes):.2f}%
Census vocab:           {len(census_genes):,}        {len(census_matched):,}           {100*len(census_matched)/len(seaad_genes):.2f}%
Brain vocab:            {len(brain_genes):,}        {len(brain_matched):,}           {100*len(brain_matched)/len(seaad_genes):.2f}%
─────────────────────────────────────────────────────────────────────
""")

# =============================================================================
# Venn diagram analysis
# =============================================================================
print("\n" + "=" * 70)
print("Step 3: Overlap analysis (Venn diagram)")
print("=" * 70)

# All three
all_three = original_matched & census_matched & brain_matched

# Exactly two
orig_census_only = (original_matched & census_matched) - brain_matched
orig_brain_only = (original_matched & brain_matched) - census_matched
census_brain_only = (census_matched & brain_matched) - original_matched

# Only one
orig_only = original_matched - census_matched - brain_matched
census_only = census_matched - original_matched - brain_matched
brain_only = brain_matched - original_matched - census_matched

print(f"""
SEAAD GENES MATCHED BY:
─────────────────────────────────────────────────────────────────────
All three vocabs:                    {len(all_three):,}
Original + Census only:              {len(orig_census_only):,}
Original + Brain only:               {len(orig_brain_only):,}
Census + Brain only:                 {len(census_brain_only):,}
Original only:                       {len(orig_only):,}
Census only:                         {len(census_only):,}
Brain only:                          {len(brain_only):,}
─────────────────────────────────────────────────────────────────────
""")

# =============================================================================
# Genes unique to Brain vocab
# =============================================================================
print("\n" + "=" * 70)
print(f"Step 4: Genes matched ONLY by Brain vocab ({len(brain_only)} genes)")
print("=" * 70)

if len(brain_only) > 0:
    brain_only_sorted = sorted(brain_only)
    print(f"First 30 genes unique to Brain vocab:")
    for i, gene in enumerate(brain_only_sorted[:30]):
        print(f"  {i+1:3d}. {gene}")
    if len(brain_only) > 30:
        print(f"  ... and {len(brain_only) - 30} more")
else:
    print("  None!")

# =============================================================================
# Genes NOT in Brain vocab but in others
# =============================================================================
print("\n" + "=" * 70)
print("Step 5: Genes MISSING from Brain vocab (but matched by others)")
print("=" * 70)

# Genes in Census but not Brain
census_not_brain = census_matched - brain_matched
print(f"\nIn Census but NOT in Brain: {len(census_not_brain):,} genes")
if len(census_not_brain) > 0:
    print(f"  Examples: {sorted(census_not_brain)[:20]}")

# Genes in Original but not Brain  
orig_not_brain = original_matched - brain_matched
print(f"\nIn Original but NOT in Brain: {len(orig_not_brain):,} genes")
if len(orig_not_brain) > 0:
    print(f"  Examples: {sorted(orig_not_brain)[:20]}")

# =============================================================================
# Create Brain vocab mapping file
# =============================================================================
print("\n" + "=" * 70)
print("Step 6: Creating Brain vocab mapping file")
print("=" * 70)

mapping_records = []
for g in seaad_genes:
    if g in brain_genes:
        mapping_records.append({
            'seaad_symbol': g,
            'scgpt_symbol': g,
            'scgpt_token_id': brain_vocab[g],
            'match_type': 'exact',
            'status': 'matched'
        })
    else:
        mapping_records.append({
            'seaad_symbol': g,
            'scgpt_symbol': None,
            'scgpt_token_id': -1,
            'match_type': None,
            'status': 'unmatched'
        })

mapping_df = pd.DataFrame(mapping_records)
mapping_df.to_csv(OUTPUT_MAPPING_PATH, index=False)
print(f"Saved to: {OUTPUT_MAPPING_PATH}")

# =============================================================================
# Final comparison table
# =============================================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(f"""
─────────────────────────────────────────────────────────────────────
                        Original        Census          Brain
─────────────────────────────────────────────────────────────────────
Vocab size:             {len(original_genes):,}          {len(census_genes):,}          {len(brain_genes):,}
SEAAD genes matched:    {len(original_matched):,}          {len(census_matched):,}          {len(brain_matched):,}
Match rate:             {100*len(original_matched)/len(seaad_genes):.2f}%          {100*len(census_matched)/len(seaad_genes):.2f}%          {100*len(brain_matched)/len(seaad_genes):.2f}%
─────────────────────────────────────────────────────────────────────
""")

# Best vocab
best_vocab = max([
    ('Original', len(original_matched)),
    ('Census', len(census_matched)),
    ('Brain', len(brain_matched))
], key=lambda x: x[1])

print(f"Best gene coverage: {best_vocab[0]} vocab with {best_vocab[1]:,} genes matched")
print("\n→ Next: Run expression analysis with Brain vocab to compare expression coverage")
print("=" * 70)