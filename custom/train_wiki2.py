# https://www.it-jim.com/blog/training-and-fine-tuning-gpt-2-and-gpt-3-models-using-hugging-face-transformers-and-openai-api/

import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import torch
import torch.utils.data
import transformers
import tqdm

TOKEN_ENDOFTEXT = 50256  # '<|endoftext|>
BLOCK_LEN = 128

# Check if CUDA (GPU) is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Specify the directory containing the .md files
wiki_dir = "kgraph.wiki"


class MyDset(torch.utils.data.Dataset):
    """A custom dataset that serves 1024-token blocks as input_ids == labels"""

    def __init__(self, data: list[list[int]]):
        self.data = []
        for d in data:
            input_ids = torch.tensor(d, dtype=torch.int64)
            attention_mask = torch.ones(len(d), dtype=torch.int64)
            self.data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids,
                }
            )

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx: int):
        data = self.data[idx]
        input_ids = data["input_ids"].clone()
        max_len = BLOCK_LEN - 1  # Account for end-of-text token
        # Pad input_ids to max_len with padding token
        input_ids = torch.nn.functional.pad(input_ids, (0, max_len - len(input_ids)), value=tokenizer.pad_token_id)
        # Shift input_ids by one to create labels predicting the next token
        labels = torch.cat((torch.tensor([TOKEN_ENDOFTEXT]), input_ids[:-1]), dim=0)
        # Create attention mask with 0 for padding and 1 for real tokens
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == tokenizer.pad_token_id] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def break_text_to_pieces(
    text: str, tokenizer: transformers.PreTrainedTokenizer, block_len: int = 512
) -> list[str]:
    """Read a file and convert it to tokenized blocks, edding <|endoftext|> to each block"""
    chunk_len0 = block_len - 1  # Leave space for a TOKEN_ENDOFTEXT
    tokens = tokenizer.encode(text)
    print(f"Number of tokens: {len(tokens)}")
    blocks = []
    pos = 0
    while pos < len(tokens):
        chunk = tokens[pos : pos + chunk_len0]
        chunk.append(TOKEN_ENDOFTEXT)
        blocks.append(chunk)
        pos += chunk_len0

    # Remove the last block if it is too short
    if len(blocks[-1]) < block_len:
        del blocks[-1]

    print(f"Number of blocks: {len(blocks)}")
    return blocks


def train_val_split(data: list[str], ratio: float):
    print(f"{len(data)}")
    n = len(data)
    assert n >= 2
    n_val = max(1, int(n * ratio))
    return data[n_val:], data[:n_val]


def prepare_dsets(
    text: str, tokenizer: transformers.PreTrainedTokenizer, block_len: int
):
    """Read the text, prepare the datasets"""
    data = break_text_to_pieces(text, tokenizer, block_len)
    data_train, data_val = train_val_split(data, 0.2)
    # train on the entire dataset
    return MyDset(data_train), MyDset(data_val)


def train_one(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
):
    """Standard PyTorch training, one epoch"""
    model.train()
    losses = []
    for batch in tqdm.tqdm(loader):
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def val_one(model: torch.nn.Module, loader: torch.utils.data.DataLoader):
    """Standard PyTorch eval, one epoch"""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            for k, v in batch.items():
                batch[k] = v.to(DEVICE)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            losses.append(loss.item())

    return np.mean(losses)


# Initialize an empty list to store the preprocessed data
preprocessed_data = []

# Iterate over all .md files in the wiki_dir directory
for filename in os.listdir(wiki_dir):
    if filename.endswith(".md"):
        file_path = os.path.join(wiki_dir, filename)
        with open(file_path, "r") as file:
            wiki_data = file.read()

        # Tokenize the text into sentences
        sentences = sent_tokenize(wiki_data)

        # Tokenize each sentence into words
        tokenized_sentences = [
            word_tokenize(sentence, preserve_line=True) for sentence in sentences
        ]

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        filtered_sentences = [
            [word for word in sentence if word.lower() not in stop_words]
            for sentence in tokenized_sentences
        ]

        # Join the filtered words back into sentences
        preprocessed_sentences = [" ".join(sentence) for sentence in filtered_sentences]

        # Print sample preprocessed_sentences
        print(
            f"Sample preprocessed sentences from file {filename}: {preprocessed_sentences[0]}"
        )

        # Append the preprocessed sentences to the preprocessed_data list
        preprocessed_data.extend(preprocessed_sentences)

        # Print size of preprocessed_data
        print(f"Current size of preprocessed_data: {len(preprocessed_data)} sentences")

# Load the pre-trained GPT-2 model and tokenizer
model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=True, padding_side="left")

# Create datasets and loader
# duplicate x times the preprocessed_data array as it's too small
preprocessed_data = preprocessed_data * 3
dset_train, dset_val = prepare_dsets(preprocessed_data, tokenizer, BLOCK_LEN)
loader_train = torch.utils.data.DataLoader(dset_train, batch_size=4)
loader_val = torch.utils.data.DataLoader(dset_val, batch_size=4)

# Optimizer, DEVICE
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
for i_epoch in range(num_epochs):
    loss_train = train_one(model, loader_train, optimizer)
    loss_val = val_one(model, loader_val)
    print(f"{i_epoch} : loss_train={loss_train}, loss_val={loss_val}")

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

# Now our model is trained, try the generation
text = "Generate a link to learn Guitar Notes Beginner Chords"
batch = tokenizer([text], return_tensors="pt")
for k, v in batch.items():
    batch[k] = v.to(DEVICE)
out = model.generate(
    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_length=20
)
print("GENERATION=", tokenizer.batch_decode(out.cpu()))
