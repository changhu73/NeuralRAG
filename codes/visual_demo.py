import torch 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import os

# embs = torch.load("C:/Users/13772/OneDrive/文档/GitHub/NeuralRAG/test/sequence_embed.pt")
chunks = torch.load("/home/Guangwei/bio/embedding/embedded_chunks.pt")

chunking_embs = []
for idx, chunk in enumerate(chunks):
    if not idx: chunk = chunk[:, 1:, :] 
    if idx == len(chunks)-1: chunk = chunk[:, :-1, :] 
    chunking_embs.append(np.mean(chunk[0], axis=0))

similarity_matrix = cosine_similarity(chunking_embs)

promoter_start = 54278193
enhancer_positions = [54675605, 54677305, 54680125, 54689965, 54695545, 54697845, 54698545, 54701285]

sequence_start = promoter_start
sequence_end = enhancer_positions[-1]
sequence_total_length = sequence_end - sequence_start

embedding_length = 2545
enhancer_indices = [(pos - sequence_start) / sequence_total_length * embedding_length for pos in enhancer_positions]
enhancer_indices = [int(idx) for idx in enhancer_indices] 

threshold = 0.84
thresholded_matrix = np.where(similarity_matrix >= threshold, similarity_matrix, 0)

first_row = thresholded_matrix[0, :]

plt.figure(figsize=(12, 2))
ax = sns.heatmap(first_row.reshape(1, -1), cmap='viridis', annot=False, cbar=True, xticklabels=False)
plt.title('Similarity Heatmap for the First Segment')
plt.xlabel('Segment Index')
plt.yticks([])

for idx in enhancer_indices:
    if 0 <= idx < first_row.shape[0]:
        plt.text(idx, 1.02, '|', color='red', fontsize=12, ha='center', va='bottom', transform=ax.get_xaxis_transform())

output_folder = "/home/Guangwei/bio/embedding"
output_filename = "similarity_heatmap.png"
output_path = os.path.join(output_folder, output_filename)

plt.savefig(output_path, format='png', bbox_inches='tight')
plt.close()