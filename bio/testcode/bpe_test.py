import sentencepiece as spm
from transformers import BertTokenizer, BertModel
import torch

sequences = ["ATCGTACGTAGCTAGCTAGCTA", "GCTAGCTAGCTAGCTAGCTAGC"]

with open('dna_sequences.txt', 'w') as f:
    for seq in sequences:
        f.write(" ".join(seq) + "\n")

spm.SentencePieceTrainer.Train('--input=dna_sequences.txt --model_prefix=dna_bpe --vocab_size=10 --character_coverage=1.0 --model_type=bpe')
sp = spm.SentencePieceProcessor()
sp.load('dna_bpe.model')
bpe_sequences = [sp.encode_as_pieces(seq) for seq in sequences]
print("BPE sequences:", bpe_sequences)
tokenizer = BertTokenizer(vocab_file='dna_bpe.vocab', do_lower_case=False)
inputs = tokenizer(bpe_sequences, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
model = BertModel.from_pretrained("zhihan1996/DNA_bert_6")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print("Embedding shape:", embeddings.shape)
