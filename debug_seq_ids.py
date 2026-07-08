from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
inputs = tokenizer("What is the capital of France?", "The capital of France is Paris.", return_tensors="pt")
seq_ids = inputs.sequence_ids()
print("seq_ids:", seq_ids)
print("tokens:", tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

mask = [s == 1 or i == 0 for i, s in enumerate(seq_ids)]
print("mask:", mask)
