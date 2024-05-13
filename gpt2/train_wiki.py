import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Specify the directory containing the .md files
wiki_dir = 'kgraph.wiki'

# Initialize an empty list to store the preprocessed data
preprocessed_data = []

# Iterate over all .md files in the wiki_dir directory
for filename in os.listdir(wiki_dir):
    if filename.endswith('.md'):
        file_path = os.path.join(wiki_dir, filename)
        with open(file_path, 'r') as file:
            wiki_data = file.read()

        # Tokenize the text into sentences
        sentences = sent_tokenize(wiki_data)

        # Tokenize each sentence into words
        tokenized_sentences = [word_tokenize(sentence, preserve_line=True) for sentence in sentences]

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_sentences = [[word for word in sentence if word.lower() not in stop_words] for sentence in tokenized_sentences]

        # Join the filtered words back into sentences
        preprocessed_sentences = [' '.join(sentence) for sentence in filtered_sentences]

        # Print sample preprocessed_sentences
        print(f"Sample preprocessed sentences from file {filename}: {preprocessed_sentences[0]}")

        # Append the preprocessed sentences to the preprocessed_data list
        preprocessed_data.extend(preprocessed_sentences)

        # Print size of preprocessed_data
        print(f"Current size of preprocessed_data: {len(preprocessed_data)} sentences")

# Split your preprocessed data into training, validation, and testing sets. This is crucial for model evaluation and hyperparameter tuning. A common split is 80% for training, 10% for validation, and 10% for testing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, preprocessed_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# Print size of X_train, X_val, X_test, y_train, y_val, y_test in one line
print(f"Size of X_train: {len(X_train)}\nSize of X_val: {len(X_val)}\nSize of X_test: {len(X_test)}\n")

# model training
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
import torch

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token

# Encode the training data
train_encodings = tokenizer(preprocessed_data, truncation=True, padding=True, return_tensors='pt')


# Create a PyTorch dataset
class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = WikiDataset(train_encodings)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2
)


# Define the training step function to calculate the loss
def training_step(model, inputs):
    outputs = model(**inputs)
    loss = outputs.loss
    if loss is not None:
        return loss
    else:
        return torch.tensor(0.0, device=device)


def compute_metrics(eval_pred):
    return {'eval_loss': eval_pred.metrics['loss']}


# Define the Trainer with the custom data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics
)

# Override the training_step method of the Trainer with the custom training_step function
trainer.training_step = training_step

# Train the model
trainer.train()

# Evaluate the model on the validation set
val_encodings = tokenizer(X_val, truncation=True, padding=True, return_tensors='pt')
val_dataset = WikiDataset(val_encodings)

print(f'Training Loss: {trainer.evaluate(train_dataset)["loss"]}')
eval_results = trainer.evaluate(val_dataset)
print(f'Validation Loss: {eval_results["loss"]}')

# Check the available keys in the eval_results dictionary
print(f"Available keys in eval_results: {eval_results.keys()}")

# Access the loss value based on the available keys
if 'eval_loss' in eval_results:
    print(f'Validation Loss: {eval_results["eval_loss"]}')
elif 'loss' in eval_results:
    print(f'Validation Loss: {eval_results["loss"]}')
else:
    print("Validation loss not found in eval_results.")

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')
