import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
from transformers import AutoTokenizer

# REFERENCE: https://huggingface.co/blog/llama2
model = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
# tokenizer = LlamaTokenizer.from_pretrained(
#             model,
#             device_map="auto")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
