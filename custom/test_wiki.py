from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine-tuned-model').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-model', padding_side='left')

# Set pad_token_id to eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

# Define the input query
query = "Return a pdf link for Guitar Beginner Chords"

# Encode the input query
query_input_ids = tokenizer.encode(query, return_tensors='pt', max_length=512, padding='max_length', truncation=True).to(device)

# Set the attention mask for the query
query_attention_mask = torch.ones_like(query_input_ids)

# Generate the response using beam search
output = model.generate(
    query_input_ids,
    attention_mask=query_attention_mask,
    max_length=1024,
    num_return_sequences=1,
    num_beams=5,
    early_stopping=True,
    return_dict_in_generate=True,
    output_scores=True,
    output_hidden_states=True,
    output_attentions=True
)

# Decode the generated response
response = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

print(f"Query: {query}")
print(f"Response: {response}")
