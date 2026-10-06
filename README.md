# Amazon-ML-challenge-2026
Business Entity Resolution - ML Challenge 2026 
# Business Entity Resolution — ML Challenge 2026

For every Source-1 (S1) business record, find the Source-2/Source-3 (S2/S3) records that refer to the same business.

**Metric:** macro F0.5 per S1 (an S1 with no true match scores 1 only if nothing is predicted).
**Data:** train covers US and India; the test set adds France with no labels.

## Results

| Split | Score (F0.5) |
| --- | --- |
| Local holdout | **0.989** |
| Public leaderboard | **0.9812** |

The ~0.008 gap between local and leaderboard comes from the test set being harder than the local holdout (unseen country, more distractor records). Competition data is not included in this repo.

## Architecture

The pipeline follows one principle: a cheap model screens every candidate, and an expensive model is used only where the cheap one is unsure.

mermaid
flowchart TD
    A["Raw records<br/>S1, S2, S3 · US / India / France"] --> B["1. Normalization<br/>accents, transliteration,<br/>legal forms, addresses"]
    B --> C["2. Entity-grouped folds<br/>E 40% · G 40% · H 20% holdout"]
    B --> D["3. Retriever<br/>char 3/4-gram embeddings<br/>InfoNCE contrastive training"]
    D --> E["Exact GPU kNN<br/>top-20 S1 per S2/S3 record"]
    E --> F["Pruning<br/>top-5 + near-ties<br/>recall 0.9937"]
    F --> G["4. Pair features<br/>~55 features"]
    G --> H["LightGBM matcher"]
    H -->|"confident French pairs<br/>(pseudo-labels)"| D
    H -->|"uncertain pairs<br/>0.02 < p < 0.98"| I["6. Cross-encoders<br/>mMiniLM · mmBERT · mmBERT"]
    H --> J["7. Monotone LightGBM stacker<br/>matcher + CE logits + context"]
    I --> J
    J --> K["8. One-record-one-S1<br/>+ F0.5 threshold"]
    K --> L["9a. Second-stage veto<br/>France only"]
    L --> M["9b. French twin filter<br/>label transfer from US/India"]
    M --> N["Output<br/>matching_results.tsv<br/>candidate_pairs.tsv"]


### 1. Normalization (`normalize.py`, `prep.py`)
Raw records are cleaned so that the same business written differently becomes comparable: accents are stripped, Indic names are transliterated, and legal forms (e.g. "Pvt Ltd", "Inc") and addresses are canonicalised.

### 2. Entity-grouped folds
Train S1 records are split by entity into folds E (40%), G (40%) and H (20%, holdout). The holdout is replicated with distractor records to resemble the test set's density.

### 3. Retriever (`block.py`)
Records are embedded using character 3/4-grams, trained with a contrastive (InfoNCE) loss. Exact GPU kNN returns the top-20 S1 candidates per S2/S3 record, which are pruned to the top-5 plus near-ties (cosine ≥ best − 0.15), keeping recall at 0.9937.

### 4. Pair features and LightGBM matcher (`features.py`, `model.py`)
For every candidate pair, ~55 features are computed: fuzzy string scores, TF-IDF similarity, number matches, name commonness, and competition (whether the record is closer to another S1). LightGBM outputs a match probability and settles roughly 97% of pairs on its own.

### 5. Pseudo-label loop for France
France has no labels, so confident French pairs from the matcher are used as pseudo-labels to retrain the retriever, which in turn yields better candidates and better pseudo-labels.

### 6. Cross-encoders (`ce.py`)
Only uncertain pairs (0.02 < p < 0.98, about 3% of all pairs) are re-scored by three cross-encoders: mMiniLM (E), mmBERT (E) and mmBERT (E+G). They read both records jointly, so they are more accurate but slower.

### 7. Stacker
A monotone LightGBM stacker combines the matcher probability, cross-encoder logits and context features into the final score.

### 8. One-record-one-S1 and thresholding
Each S2/S3 record may match at most one S1 record. After enforcing this, a threshold is chosen to maximise F0.5.

### 9. France-only post-processing
The cross-encoders agreed with the matcher far less on French data (85% vs 99%+ in US/India), so France keeps the matcher's probabilities. Two filters then remove false matches:
- **Second stage (`stage2.py`):** a veto-only model that removes doubtful matches.
- **French twin filter (`twins.py`):** uses label transfer from US/India. Where France accepted "same address, different organisation word" matches far more often than the natural rate in US/India, the excess were different businesses ("twins") and were removed.

## Key takeaways

1. **Validation must look like the test.** A standard holdout over-predicts the leaderboard, so the holdout was built with test-like distractor density and entity-grouped folds.
2. **Cheap model first, expensive model only where unsure.** LightGBM handles ~97% of pairs; cross-encoders on the uncertain ~3% gave the largest single gain.
3. **Under F0.5, removing doubtful matches beats adding uncertain ones.** Every removal rule (one-record-one-S1, veto, twin filter) transferred to the test set.
4. **"Multilingual" does not mean it works on a new country.** A label-free agreement check caught the cross-encoders' weakness on French before submission.

## Repository structure

```
student_resource/code/business_entity_resolution/   # reproducible pipeline
tools/                                              # helper scripts
```
