"""
Since fine-tuning GPT-2 might not be the most suitable approach for your data due to its mixed content and limited size, here are alternative methods to answer custom questions using your preprocessed data:

1. Retrieval-Based Question Answering:

This approach retrieves relevant documents or passages from your data that best answer the user's question. Here's how it works:

Data Preprocessing:
Tokenize your text data (split into words/subwords).
Create an inverted index: This data structure maps words/subwords to the documents they appear in.
Question Answering:
Tokenize the user's question.
Use the inverted index to find documents containing the question words.
Use a ranking algorithm (e.g., TF-IDF) to score the retrieved documents based on their relevance to the question.
Return the top-ranked document(s) or a summary as the answer.
Libraries/Tools:

scikit-learn (for TF-IDF)
Haystack (a comprehensive library for information retrieval)
2. Dense Passage Retrieval with Transformers:

This approach uses pre-trained sentence encoders like transformers to embed both questions and your documents into high-dimensional vectors. The similarity between question and document vectors determines the relevance.

Data Preprocessing:
Encode each document into a vector representation using a transformer model (e.g., Sentence Transformers).
Question Answering:
Encode the user's question into a vector representation.
Calculate the cosine similarity between the question vector and each document vector.
Return the document(s) with the highest similarity score(s) as the answer.
Libraries/Tools:

Sentence Transformers
Faiss (for efficient similarity search)
3. Combining Retrieval and Generative Models (Hybrid Approach):

This method combines retrieval with a generative model like a fine-tuned GPT-2. Here's the process:

Retrieval:
Use one of the methods above to retrieve a few relevant documents.
Generative Model (Optional):
Fine-tune a GPT-2 model on your retrieved documents (if data allows).
Use the fine-tuned model to refine the answer by generating a summary or answer text based on the retrieved documents.
Benefits:

Retrieval ensures relevant information is retrieved even with limited data.
Generative models can improve answer quality by summarizing or providing more informative responses.
Considerations:

This approach requires more complex implementation, potentially needing fine-tuning GPT-2 if you want a generative component.
Choosing the Best Approach:

The most suitable method depends on your specific needs and data size.

For simple questions and limited data, retrieval-based approaches are effective.
If answer quality and generating summaries are critical, explore dense retrieval with transformers or a hybrid approach.
Remember that you'll need to implement the chosen method with suitable libraries and tailor it to your specific data format (tokenization needed). If you're new to these techniques, starting with simpler retrieval-based approaches might be easier.

"""

"""
Explanation:
Sentence Transformer: We use SentenceTransformer to encode questions and documents into high-dimensional vectors.
Preprocess Data: This function tokenizes documents and builds an inverted index to efficiently find documents containing specific words.
Answer Question: This function takes a question, encodes it, and retrieves documents relevant to the question words using the inverted index. It then calculates the cosine similarity between the question vector and each document vector to rank them. Finally, it returns the top-ranked documents as the answer.
"""

import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch


# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Specify the directory containing the .md files
wiki_dir = "kgraph.wiki"


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


# Load Sentence Transformer model (you'll need to install it)
model = SentenceTransformer("all-mpnet-base-v2")


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


# Example usage
question = "What are some best practices for writing Dockerfiles?"
answers = answer_question(question)
print("Answers:")
for answer in answers:
    print(answer)
