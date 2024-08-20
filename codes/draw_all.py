import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

txt_folder = "data/DNA_sequence/output_files_Fulco_3a"
pt_folder = "data/embedded_dataset/output_files_Fulco_3a"
output_folder = "data/graph/output_files_Fulco_3a"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_groups = defaultdict(list)

for filename in os.listdir(txt_folder):
    if filename.endswith(".txt"):
        parts = filename[:-4].split("_")
        chr_num = parts[0]
        promoter_start = int(parts[1])
        enhancer_position = int(parts[2])
        file_groups[(chr_num, promoter_start)].append(enhancer_position)

for (chr_num, promoter_start), enhancer_positions in file_groups.items():
    pt_filename = f"{chr_num}_{promoter_start}_{max(enhancer_positions)}_embedded_chunks.pt"
    pt_filepath = os.path.join(pt_folder, pt_filename)
    
    if not os.path.exists(pt_filepath):
        print(f"PT file not found: {pt_filepath}")
        continue
    
    chunks = torch.load(pt_filepath)

    chunking_embs = []
    for idx, chunk in enumerate(chunks):
        if not idx: 
            chunk = chunk[:, 1:, :] 
        if idx == len(chunks) - 1: 
            chunk = chunk[:, :-1, :]
        chunking_embs.append(np.mean(chunk[0], axis=0))

    similarity_matrix = cosine_similarity(chunking_embs)

    sequence_start = promoter_start
    sequence_end = max(enhancer_positions)
    sequence_total_length = sequence_end - sequence_start

    embedding_length = len(chunking_embs)
    enhancer_indices = [(pos - sequence_start) / sequence_total_length * (embedding_length - 1) for pos in enhancer_positions]
    enhancer_indices = [int(idx) for idx in enhancer_indices]

    threshold = 0.84
    thresholded_matrix = np.where(similarity_matrix >= threshold, similarity_matrix, 0)

    first_row = thresholded_matrix[0, :]

    plt.figure(figsize=(12, 2))
    ax = sns.heatmap(first_row.reshape(1, -1), cmap='viridis', annot=False, cbar=True, xticklabels=False)
    plt.title(f'Similarity Heatmap for {chr_num}_{promoter_start}')
    plt.xlabel('Segment Index')
    plt.yticks([])

    for idx in enhancer_indices:
        if 0 <= idx < first_row.shape[0]:
            plt.text(idx, 1.02, '|', color='red', fontsize=12, ha='center', va='bottom', transform=ax.get_xaxis_transform())

    output_filename = f"{chr_num}_{promoter_start}_similarity_heatmap.png"
    output_path = os.path.join(output_folder, output_filename)

    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to {output_path}")
