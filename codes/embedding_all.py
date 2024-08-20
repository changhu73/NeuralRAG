import os
import torch
from embedders import DNABert2Embedder

def split_dna_sequence(sequence, segment_length=128):
    return [sequence[i:i+segment_length] for i in range(0, len(sequence) - segment_length + 1, segment_length)]

def embed_dna_sequence(file_path, embedder, output_folder):
    with open(file_path, 'r') as f:
        dna_sequence = f.readline().strip()

    print(f'Processing file: {file_path}')
    print(f'Ori DNA length: {len(dna_sequence)}')

    sequence_embed, embedded_chunks = embedder(dna_sequence, upsample_embeddings=True)
    
    print(f'Number of embedded chunks: {len(embedded_chunks)}')
    print(f'Shape of each chunk: {embedded_chunks[0].shape}')
    
    filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]
    
    output_path = os.path.join(output_folder, f'{filename_without_extension}_embedded_chunks.pt')
    
    torch.save(embedded_chunks, output_path)
    print(f'Saved embeddings to {output_path}\n')

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    embedder = DNABert2Embedder()
    embedder.max_length = 128

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            embed_dna_sequence(file_path, embedder, output_folder)

if __name__ == "__main__":
    input_folder = 'data/DNA_sequence/output_files_Fulco_3a' 
    output_folder = 'data/embedded_dataset/output_files_Fulco_3a'
    
    process_folder(input_folder, output_folder)
