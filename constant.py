import os

DATA_DIR = "./mimicdata/"
MIMIC_3_DIR = os.path.join(DATA_DIR, "mimic3")
MIMIC_4_DIR = os.path.join(DATA_DIR, "mimic4_9")
MIMIC_4_10_DIR = os.path.join(DATA_DIR, "mimic4_10")
UMLS_PATH = DATA_DIR

FACET_DIMENSIONS = ['Definition', 'Symptoms', 'Diagnosis', 'Treatment', 'Complications', 'Summary']
K_FACETS = 6

LORA_RANK = 2
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

BM25_K1 = 1.2
BM25_B = 0.75
TOP_K_ABSTRACTS = 3

PARAM_GEN_HIDDEN = 1024
PARAM_GEN_POS_ENC = 64

DISTILLATION_LAMBDA = 0.01

ICD_HIERARCHY_DEPTH = 2
SYNONYM_CAP = 6
RELATION_NEIGHBOR_CAP = 10