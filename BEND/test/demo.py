import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
# from scipy.cluster.hierarchy import linkage, dendrogram
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

from embedders import DNABert2Embedder


def split_dna_sequence(sequence, segment_length=128):
    return [sequence[i:i+segment_length] for i in range(0, len(sequence) - segment_length + 1, segment_length)]


with open('/home/guangwei/bio/BEND/test/demoDNA.txt') as f:
    dna_sequence = f.readline()
print('Ori DNA length: ', len(dna_sequence)) # 325740

segments = split_dna_sequence(dna_sequence, segment_length=128)
# embeddings = []

embedder = DNABert2Embedder()
embedder.max_length = 128


first_para, embedded_chunks = embedder.embed(dna_sequence, upsample_embeddings=True)


# sequence_embed = embedder(segments, upsample_embeddings = True,)
# print(sequence_embed.shape)
    # # embedding with mean poolingembeddings.append(torch.mean(hidden_states[0], dim=0).detach().numpy()) # expect to be 768
# torch.save(sequence_embed, 'sequence_embed.pt')

torch.save(embedded_chunks, '/home/guangwei/bio/BEND/test/embedded_chunks.pt')

