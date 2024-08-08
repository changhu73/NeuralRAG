from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
model = BertModel.from_pretrained("zhihan1996/DNA_bert_6")

sequence = "ATCGTACGTAGCTAGCTAGCTA"

k = 6
kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

inputs = tokenizer(" ".join(kmers), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print("Embedding shape:", embeddings.shape)
