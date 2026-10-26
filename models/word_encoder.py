import torch
from torch import nn
import numpy as np
import gensim


class Word_Encoder(nn.Module):
    def __init__(self, word_config={}):
        super(Word_Encoder, self).__init__()
        self.word_config = word_config
        self.count = word_config['count']
        self.dim = word_config['dim']
        self.dropout = word_config.get('dropout', 0.0)
        self.frz = word_config.get('frz', False)

        self.embedding = nn.Embedding(self.count, self.dim, padding_idx=word_config.get('padding_idx', None))
        self.dropout_layer = nn.Dropout(self.dropout)

        word_embedding_path = word_config.get('word_embedding_path', None)
        if word_embedding_path:
            self.load_word_embedding(word_embedding_path)

        if self.frz:
            for p in self.embedding.parameters():
                p.requires_grad = False

    def load_word_embedding(self, path):
        if path.endswith('.model'):
            model = gensim.models.Word2Vec.load(path)
            word_vectors = model.wv
            word_count = len(word_vectors)
            embeddings = np.zeros((self.count, self.dim))
            loaded = 0
            for i in range(self.count):
                word = model.wv.index2word[i] if hasattr(model.wv, 'index2word') else None
                if word is None:
                    continue
                try:
                    embeddings[i] = word_vectors[word]
                    loaded += 1
                except:
                    pass
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
            print(f"Loaded {loaded} word embeddings from {path}")
        elif path.endswith('.bin') or path.endswith('.txt'):
            try:
                word_vectors = gensim.models.KeyedVectors.load_word2vec_format(path, binary=path.endswith('.bin'))
                embeddings = np.zeros((self.count, self.dim))
                for i in range(self.count):
                    word = word_vectors.index2word[i] if hasattr(word_vectors, 'index2word') else None
                    if word is None:
                        continue
                    try:
                        embeddings[i] = word_vectors[word]
                    except:
                        pass
                self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
            except:
                pass

    def forward(self, input_word):
        embedded = self.embedding(input_word)
        embedded = self.dropout_layer(embedded)
        return embedded