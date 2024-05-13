import torch
import transformers

# Check if CUDA (GPU) is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLOCK_LEN = 512
TOKEN_ENDOFTEXT = 50256  # '<|endoftext|>

# Load the fine-tuned model and tokenizer
model = transformers.GPT2LMHeadModel.from_pretrained("./fine-tuned-model").to(DEVICE)
tokenizer = transformers.GPT2Tokenizer.from_pretrained("./fine-tuned-model", padding_side="left")
# model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
# tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")

# Add a padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Text prompt for generation
text = ["Generate a link to learn Guitar Notes Beginner Chords"]

# Tokenize the text prompt
batch = tokenizer(text, return_tensors="pt", max_length=BLOCK_LEN, padding="max_length", truncation=True)

# Move the batch to the appropriate device
for k, v in batch.items():
    batch[k] = v.to(DEVICE)

# Generate text based on the input prompt
output = model.generate(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    max_length=BLOCK_LEN*2,  # Increase max_length for longer text
    num_return_sequences=1,
    no_repeat_ngram_size=3,  # Adjust n-gram size for diversity
    top_k=50,
    top_p=0.95,
    temperature=0.8,  # Slightly increase temperature for more randomness
    eos_token_id=TOKEN_ENDOFTEXT,
    do_sample=True,
)

# Decode and print the generated text
generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
print("Generated Text:", generated_text)

enc = tokenizer(text, return_tensors='pt')
input_ids = enc['input_ids']

# Add one token at a time, in a loop
for i in range(20):
    # print_it(input_ids, 'input_ids')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.int64)
    logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
    new_id = logits[:, -1, :].argmax(dim=1)       # Generate new ID
    input_ids = torch.cat([input_ids, new_id.unsqueeze(0)], dim=1)   # Add new token

print(tokenizer.batch_decode(input_ids))  # Decode result
