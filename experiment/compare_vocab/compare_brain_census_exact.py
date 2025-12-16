#!/usr/bin/env python3
"""
Check if Brain vocab is exactly the same as Census vocab
"""

import json

# =============================================================================
# PATHS
# =============================================================================
CENSUS_VOCAB_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/scGPT/scgpt/tokenizer/default_census_vocab.json"
BRAIN_VOCAB_PATH = "/home/ye/ml-experiments/zhaozhuoresearch/quiz/research-project-gene-ml/scGPT/examples/save/scGPT_brain/vocab.json"

# =============================================================================
# Load vocabularies
# =============================================================================
print("=" * 70)
print("Loading vocabularies")
print("=" * 70)

with open(CENSUS_VOCAB_PATH, 'r') as f:
    census_vocab = json.load(f)

with open(BRAIN_VOCAB_PATH, 'r') as f:
    brain_vocab = json.load(f)

print(f"Census vocab size: {len(census_vocab)}")
print(f"Brain vocab size:  {len(brain_vocab)}")

# =============================================================================
# Compare keys (gene names)
# =============================================================================
print("\n" + "=" * 70)
print("Comparing keys (gene names)")
print("=" * 70)

census_keys = set(census_vocab.keys())
brain_keys = set(brain_vocab.keys())

shared_keys = census_keys & brain_keys
census_only_keys = census_keys - brain_keys
brain_only_keys = brain_keys - census_keys

print(f"Shared keys:      {len(shared_keys)}")
print(f"Census-only keys: {len(census_only_keys)}")
print(f"Brain-only keys:  {len(brain_only_keys)}")

if census_only_keys:
    print(f"\nKeys in Census but not Brain: {sorted(census_only_keys)[:20]}")
if brain_only_keys:
    print(f"\nKeys in Brain but not Census: {sorted(brain_only_keys)[:20]}")

# =============================================================================
# Compare values (token IDs) for shared keys
# =============================================================================
print("\n" + "=" * 70)
print("Comparing values (token IDs) for shared keys")
print("=" * 70)

mismatched_ids = []
for key in shared_keys:
    if census_vocab[key] != brain_vocab[key]:
        mismatched_ids.append((key, census_vocab[key], brain_vocab[key]))

print(f"Keys with same token ID:      {len(shared_keys) - len(mismatched_ids)}")
print(f"Keys with different token ID: {len(mismatched_ids)}")

if mismatched_ids:
    print(f"\nFirst 20 mismatched IDs:")
    for key, census_id, brain_id in mismatched_ids[:20]:
        print(f"  {key}: Census={census_id}, Brain={brain_id}")

# =============================================================================
# Final verdict
# =============================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if census_keys == brain_keys and len(mismatched_ids) == 0:
    print("✓ IDENTICAL: Brain vocab is exactly the same as Census vocab")
elif census_keys == brain_keys:
    print(f"⚠ SAME KEYS, DIFFERENT IDS: {len(mismatched_ids)} genes have different token IDs")
else:
    print(f"✗ DIFFERENT VOCABS:")
    print(f"  - {len(census_only_keys)} genes only in Census")
    print(f"  - {len(brain_only_keys)} genes only in Brain")
    print(f"  - {len(mismatched_ids)} shared genes have different IDs")

print("=" * 70)