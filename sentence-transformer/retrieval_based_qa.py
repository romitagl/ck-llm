import os
import pickle
from wiki_processor import process_wiki_dir
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch


def load_cached_data(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def save_cached_data(data, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)


cache_file = 'preprocessed_data.pkl'

cached_data = load_cached_data(cache_file)
if cached_data is None:
    preprocessed_data = process_wiki_dir()  # Process the wiki directory to get preprocessed data
    save_cached_data(preprocessed_data, cache_file)
else:
    preprocessed_data = cached_data


# Preprocess your data
def preprocess_data(data):
    documents = []
    inverted_index = defaultdict(list)
    for doc in data:
        text = doc
        documents.append(text)
        tokens = text.lower().split()  # Tokenize and lowercase
        for token in tokens:
            inverted_index[token].append(documents.index(text))
    return documents, inverted_index


# Load your preprocessed data
documents, inverted_index = preprocess_data(preprocessed_data)

# Load Sentence Transformer model (you'll need to install it)
model = SentenceTransformer("all-mpnet-base-v2")


def answer_question(question):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Get the available device
    question_embedding = model.encode(question, convert_to_tensor=True).to(
        device
    )  # Move to the device
    relevant_docs = []
    for word in question.lower().split():
        relevant_docs.extend(inverted_index.get(word, []))
    relevant_docs = set(relevant_docs)  # Remove duplicates

    # Rank documents based on cosine similarity (higher score = more relevant)
    doc_scores = {
        doc: util.pytorch_cos_sim(
            question_embedding, torch.tensor(model.encode(documents[doc])).to(device)
        )  # Convert to PyTorch tensor and move to the device
        for doc in relevant_docs
    }
    top_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[
        :1
    ]  # Return top 1

    # Return top documents or their summaries (modify based on your needs)
    return [documents[doc_id] for doc_id, _ in top_docs]
