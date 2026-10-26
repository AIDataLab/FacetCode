import os
import json
import pickle
import numpy as np


def coder_init(code_embedding_path, ind2c, dim, device, version=None):
    if not os.path.exists(code_embedding_path):
        print(f"Code embedding file not found: {code_embedding_path}")
        return None

    try:
        if code_embedding_path.endswith('.json'):
            with open(code_embedding_path, 'r') as f:
                code_embs = json.load(f)
        elif code_embedding_path.endswith('.pkl'):
            with open(code_embedding_path, 'rb') as f:
                code_embs = pickle.load(f)
        else:
            return None

        code_embs_dict = {}
        for code, emb in code_embs.items():
            code_embs_dict[code] = np.array(emb)

        result = {}
        for i in ind2c:
            code = ind2c[i]
            if code in code_embs_dict:
                result[code] = code_embs_dict[code]
            else:
                result[code] = np.zeros(dim)

        return result
    except Exception as e:
        print(f"Error loading code embeddings: {e}")
        return None


def load_code_embedding(code_embedding_path):
    if not os.path.exists(code_embedding_path):
        return None

    try:
        if code_embedding_path.endswith('.json'):
            with open(code_embedding_path, 'r') as f:
                return json.load(f)
        elif code_embedding_path.endswith('.pkl'):
            with open(code_embedding_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading code embedding: {e}")
        return None