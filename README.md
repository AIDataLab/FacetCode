# FacetCode

Multi-Faceted Code Profiling for Automatic ICD Coding

## Overview

FacetCode is a framework for automatic ICD (International Classification of Diseases) coding that leverages multi-faceted disease profiling and parameter-efficient knowledge injection. Unlike traditional methods that represent each ICD code as a single monolithic vector, FacetCode decomposes medical knowledge into orthogonal clinical dimensions and internalizes structured knowledge directly into model parameters through ICD-aware parameter generation.

## Method Overview

### Multi-Faceted Knowledge Parameterization

1. **Facet-oriented Knowledge Construction**: Build structural query anchors from ICD hierarchy topology, expand via UMLS graph traversal to extract synonyms and clinical relations, retrieve PubMed abstracts via BM25.

2. **Code Parameterization via LoRA**: Fine-tune a base LLM with LoRA independently for each ICD code, learning low-rank adapter matrices that encode code-specific knowledge.

3. **ICD-aware Parameter Generator**: Train a hypernetwork (MLP) that maps code descriptions to their corresponding LoRA adapter parameters, enabling knowledge sharing across the ICD taxonomy.

### Hierarchy-Refined Multi-Dimensional Classification

1. **Multi-dimensional Semantic Expansion**: Generate structured descriptions covering 6 clinical dimensions (Definition, Symptoms, Diagnosis, Treatment, Complications, Summary) using the adapted model with injected parameters.

2. **Hierarchy-Guided Attention**: Propagate ancestral context into code representations using attention mechanism based on Summary dimension affinity.

3. **Cross-Facet Evidence Fusion**: Mutual attention mechanism that aligns clinical evidence from documents with corresponding code facets, producing final classification predictions.

## Architecture

The framework consists of:

- **ICDParameterGenerator**: Hypernetwork that predicts code-specific LoRA adapters on demand
- **FacetEncoder**: BiLSTM-based encoder for multi-dimensional code representations
- **HierarchyGuidedAttention**: Attention module for taxonomic knowledge propagation
- **CrossFacetFusion**: Channel attention mechanism for evidence alignment

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

Obtain licenses to download MIMIC-III and MIMIC-IV datasets.

Follow the preprocessing steps from CAML-MIMIC and ICD-MSMN to obtain the final JSON format files.

## Word Embeddings

Download pre-trained Word2Vec embeddings (e.g., from LAAT repository) and update the path in configuration.

## Training

MIMIC-III Full (4 GPUs):

```bash
accelerate launch --multi_gpu main.py \
    --version mimic3 \
    --combiner lstm \
    --decoder FacetCodeV1 \
    --attention_head 1 \
    --attention_head_dim 256 \
    --attention_dim 512 \
    --learning_rate 5e-4 \
    --train_epoch 30 \
    --batch_size 4 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --xavier \
    --main_code_loss_weight 0.0 \
    --rdrop_alpha 5.0 \
    --est_cls 1 \
    --term_count 8 \
    --word_embedding_path embedding/word2vec_sg0_100.model \
    --head_pooling mean \
    --text_pooling max \
    --alpha_weight 0.01 \
    --use_graph \
    --topk_num 300 \
    --output_base_dir ./outputs/
```

MIMIC-III 50:

```bash
accelerate launch main.py \
    --version mimic3-50 \
    --combiner lstm \
    --decoder FacetCodeV1 \
    --attention_head 1 \
    --attention_head_dim 256 \
    --attention_dim 512 \
    --learning_rate 5e-4 \
    --train_epoch 40 \
    --batch_size 16 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --xavier \
    --main_code_loss_weight 0.0 \
    --rdrop_alpha 12.5 \
    --est_cls 1 \
    --term_count 8 \
    --word_embedding_path embedding/word2vec_sg0_100.model \
    --head_pooling mean \
    --text_pooling max \
    --alpha_weight 0.001 \
    --use_graph \
    --output_base_dir ./outputs/
```

MIMIC-III rare50 (long-tailed codes with <50 training samples):

```bash
accelerate launch main.py \
    --version mimic3-rare50 \
    --combiner lstm \
    --decoder FacetCodeV1 \
    --attention_head 1 \
    --attention_head_dim 256 \
    --attention_dim 512 \
    --learning_rate 5e-4 \
    --train_epoch 50 \
    --batch_size 8 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --xavier \
    --main_code_loss_weight 0.0 \
    --rdrop_alpha 5.0 \
    --est_cls 1 \
    --term_count 8 \
    --word_embedding_path embedding/word2vec_sg0_100.model \
    --head_pooling mean \
    --text_pooling max \
    --alpha_weight 0.01 \
    --use_graph \
    --topk_num 100 \
    --use_facet_code \
    --k_facets 6 \
    --param_gen_hidden 1024 \
    --lora_rank 2 \
    --output_base_dir ./outputs/
```

## Supported Datasets

- MIMIC-III-Full (mimic3)
- MIMIC-III-50 (mimic3-50)
- MIMIC-III-rare50 (mimic3-rare50)
- MIMIC-IV-ICD9-Full (mimic4)
- MIMIC-IV-ICD9-50 (mimic4-50)
- MIMIC-IV-ICD10-Full (mimic4_10)
- MIMIC-IV-ICD10-50 (mimic4_10-50)

## Key Parameters

- `--use_facet_code`: Enable multi-faceted code profiling
- `--k_facets`: Number of clinical dimensions (default: 6)
- `--param_gen_hidden`: Parameter generator hidden dimension
- `--lora_rank`: LoRA rank for code-specific adaptation
- `--use_graph`: Enable graph-based code relation modeling
