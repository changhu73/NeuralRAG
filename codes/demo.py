import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
# from scipy.cluster.hierarchy import linkage, dendrogram
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

from codes.embedders import DNABert2Embedder


def split_dna_sequence(sequence, segment_length=128):
    return [sequence[i:i+segment_length] for i in range(0, len(sequence) - segment_length + 1, segment_length)]


with open('/home/Guangwei/bio/embedding/demoDNA.txt') as f:
    dna_sequence = f.readline()
print('Ori DNA length: ', len(dna_sequence)) # 325740

embedder = DNABert2Embedder()
embedder.max_length = 128

sequence_embed, embedded_chunks = embedder(dna_sequence, upsample_embeddings = True,)
print(len(embedded_chunks))
print(embedded_chunks[0].shape)

torch.save(embedded_chunks, '/home/Guangwei/bio/embedding/embedded_chunks.pt')

