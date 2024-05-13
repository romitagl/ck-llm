# ck-llm

Custom Knowledge LLM - Use a base LLM and infuse structured knowledge

## Models

### Retrieval-Based Question Answering

This approach retrieves relevant documents from my Wiki repository: [wiki](https://github.com/romitagl/kgraph.wiki.git).

Sentence Transformer:
We use the SentenceTransformer library and the "all-mpnet-base-v2" model to encode questions and documents into high-dimensional vectors.
The SentenceTransformer library is a powerful tool that can be used for a variety of natural language processing tasks, including question answering.
The "all-mpnet-base-v2" model is a pre-trained language model that is specifically designed for question answering.
When we encode a question or document using this model, we get a fixed-length vector that captures the meaning of the input.

Preprocess Data:
This function tokenizes documents and builds an inverted index to efficiently find documents containing specific words.
Tokenizing involves breaking the text into individual words or subwords.
The inverted index is a data structure that maps each word or subword to a set of documents that contain that word or subword.
This allows us to quickly find documents that contain specific words or phrases when answering a question.

Answer Question:
This function takes a question, encodes it, and retrieves documents relevant to the question words using the inverted index.
It works as follows:

1. Tokenize the question: Break the question into individual words or subwords.
2. Encode the question: Use the SentenceTransformer model to encode the question into a fixed-length vector.
3. Find relevant documents: Use the inverted index to find documents that contain any of the question words or subwords.
4. Calculate cosine similarity: For each retrieved document, calculate the cosine similarity between the question vector and the document vector.
5. Rank documents: Rank the documents by their cosine similarity score.
6. Return top-ranked documents: Return the top-ranked documents as the answer.

The end result is that we can use our custom LLM to answer questions based on the content of my Wiki repository.

#### Steps

**Clone the Wiki Repository**: Start by cloning your Wiki repository from GitHub using the command `git clone https://github.com/romitagl/kgraph.wiki.git`. This will download the repository to your local machine, allowing you to access and manipulate the data.

   ```bash
   cd custom
   git clone https://github.com/romitagl/kgraph.wiki.git
   ```

**Preprocessing and Data Preparation**: Next, you'll need to preprocess and prepare your data for training the LLM. This typically involves tokenizing the text, removing stop words, and converting the data into a format suitable for model training. You can use libraries like NLTK or spaCy for tokenization and preprocessing.

   ```python
   import os
   import nltk
   from nltk.corpus import stopwords
   from nltk.tokenize import word_tokenize, sent_tokenize

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
   ```

   Code above reads the Wiki data from the `kgraph.wiki` .md files, tokenizes it into sentences, tokenizes each sentence into words, removes stop words, and joins the filtered words back into sentences.

**Data Preprocessing**: Tokenize your text data (split into words/subwords). Create an inverted index: This data structure maps words/subwords to the documents they appear in.

**Question Answering**: Tokenize the user's question. Use the inverted index to find documents containing the question words.
Use a ranking algorithm (e.g., TF-IDF) to score the retrieved documents based on their relevance to the question.
Return the top-ranked document(s) or a summary as the answer.

### Llama-2-7b

```bash
# install huggingface-cli (https://huggingface.co/docs/huggingface-cli) if not installed
pip install huggingface_hub
huggingface-cli login
```

```bash
cd llama-2-7b-hf
# install dependencies
python -m pip install -r requirements.txt 
python text-completion.py
```

Run in a `C` program: <https://github.com/karpathy/llama2.c>/.

#### Notes

LlamaTokenizer requires the SentencePiece library. Checkout the instructions on the
installation page of its repo: <https://github.com/google/sentencepiece#installation>.

On Mac, install the sentencepiece library with: `brew install sentencepiece`.

### TinyStories-33M

Let's see how to finetune a language model to generate character backstories using HuggingFace Trainer with wandb integration. We'll use a tiny language model (TinyStories-33M) due to resource constraints: [./tinystories-33m/05_train_llm_starter.ipynb](tinystories-33m/05_train_llm_starter.ipynb)

## Description

Traditionally, a base model is trained with point-in-time data to ensure its effectiveness in performing specific tasks and adapting to the desired domain. However, sometimes you need to work with newer or more current data.
Two approaches can supplement the base model:

- fine-tuning or further training of the base model with new data
- RAG that uses prompt engineering to supplement or guide the model in real time.

Fine-tuning is suitable for continuous domain adaptation, enabling significant improvements in model quality but often incurring higher costs.
Conversely, RAG offers an alternative approach, allowing the use of the same model as a reasoning engine over new data provided in a prompt.
This technique enables in-context learning without the need for expensive fine-tuning, empowering businesses to use LLMs more efficiently.

### Training

Training a language model to answer questions using an unsupervised dataset composed of articles downloaded from the internet involves several steps. Here's a general outline of the process:

### Step 1: Data Collection and Preprocessing

Data Collection: Download a large corpus of articles from the internet. This can be done using web scraping techniques or by leveraging publicly available datasets like Common Crawl.

Preprocessing:

- Clean the text data by removing HTML tags, punctuation, and special characters.
- Tokenize the text into individual words or subwords (smaller units of words).

This step is crucial for the model to understand the structure of the text.

### Step 2: Model Selection and Training

Model Selection: Choose a suitable language model architecture, such as a transformer-based model, which is known for its ability to process long-range dependencies in text.

Training: Train the model on the preprocessed dataset using a self-supervised learning objective, such as masked language modeling or next sentence prediction.
This step helps the model learn the patterns and relationships within the text without explicit supervision.

### Step 3: Question Answering

Question Embedding: Convert the question into a numerical representation (embedding) that can be processed by the model.
This can be done using techniques like word embeddings (e.g., Word2Vec, GloVe) or more advanced methods like BERT.

Answer Generation: Use the trained language model to generate a response to the question.
This can be done by predicting the next word in a sequence given the question and the context from the training data.

### Step 4: Evaluation and Fine-Tuning (Optional)

Evaluation: Assess the performance of the model on a test set of questions.
This can be done using metrics like accuracy, F1 score, or mean average precision.

Fine-Tuning: If the model's performance is not satisfactory, fine-tune the model on a small labeled dataset specific to the question answering task.
This step can significantly improve the model's performance on the target task.

### Step 5: Deployment

Deployment: Once the model is trained and evaluated, deploy it in a production environment where it can be used to answer questions.
This can be done through APIs, chatbots, or other interfaces.

### Additional Considerations

Data Quality: Ensure that the dataset is diverse and representative of the internet articles.
This can be achieved by using techniques like data augmentation or filtering out low-quality content.

Model Size and Complexity: Balance the model's size and complexity to achieve a good trade-off between performance and computational resources.

Regularization Techniques: Apply regularization techniques like dropout or weight decay to prevent overfitting and improve the model's generalizability.

### Fine-tuning

Fine-tuning a base foundation Generative Pre-trained Transformer (GPT) model involves taking a pre-trained model and adapting it to a specific task or domain using task-specific data. In this case, we'll assume you have a README.md file that contains relevant information for your task. Below is a step-by-step guide on how to use the GPT model and fine-tune it using knowledge from the README.md file:

1. **Set Up Environment and Data:**
   - Install the required libraries and frameworks for working with the GPT model (e.g., Transformers library from Hugging Face).
   - Prepare your task-specific data, including the README.md file and any associated labels or targets.

2. **Load Pre-trained GPT Model:**
   - Select the base foundation GPT model you want to use (e.g., [BLOOM](https://bigscience.huggingface.co/blog/bloom), [LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)) and load it using the appropriate library (e.g., [Hugging Face's Transformers library](https://huggingface.co/docs/transformers/index)).

3. **Tokenize the Data:**
   - Tokenize the text data from the README.md file using the tokenizer that corresponds to the pre-trained GPT model. This step converts the text into a format the model can understand, typically a sequence of integer tokens.

4. **Create Data Loaders:**
   - Organize your tokenized data into data loaders or data batches, depending on the framework you're using (e.g., PyTorch or TensorFlow).

5. **Define Task-Specific Head:**
   - To adapt the GPT model to your specific task, you need to add a task-specific head on top of the base model. The head is typically a neural network layer tailored to your task (e.g., a linear layer for text classification, or a language model head for text generation).

6. **Initialize Parameters:**
   - Initialize the parameters of the task-specific head with random values. The rest of the pre-trained GPT model parameters will retain their pre-trained values.

7. **Define Fine-tuning Objective:**
   - Set up the fine-tuning objective for your specific task. For instance, if it's a classification task, you may use cross-entropy loss, while for a language modeling task, you may use masked language modeling loss.

8. **Fine-tuning Process:**
   - Start fine-tuning the model on your task-specific data using the task-specific head. Feed the tokenized data into the model, compute the loss, and backpropagate the gradients to update the model's parameters.

9. **Hyperparameter Tuning:**
   - Fine-tuning typically requires adjusting hyperparameters like learning rate, batch size, and the number of training epochs. Experiment with different values to find the best performing setup for your task.

10. **Validation and Evaluation:**
    - During fine-tuning, use a validation set to monitor the model's performance and prevent overfitting. Evaluate the fine-tuned model on a separate test set to assess its generalization.

11. **Iterate and Refine:**
    - Fine-tuning is an iterative process. If the performance is not satisfactory, you can experiment with different architectures, hyperparameters, or data augmentation techniques to further improve the model's performance.

Remember that fine-tuning a GPT model may require significant computational resources and time, especially for larger models like GPT-3. Additionally, be mindful of licensing restrictions if you are using a pre-trained model provided by a third-party platform. Always review and comply with the terms and conditions for fine-tuning and deploying the model in your application.

### RAG

Source: <https://learn.microsoft.com/en-us/azure/machine-learning/concept-retrieval-augmented-generation>

Retrieval Augmented Generation (RAG) is a pattern that works with pretrained Large Language Models (LLM) and your own data to generate responses.

In information retrieval, RAG is an approach that enables you to harness the power of LLMs with your own data.
Enabling an LLM to access custom data involves the following steps.
First, the large data should be chunked into manageable pieces.
Second, the chunks need to be converted into a searchable format.
Third, the converted data should be stored in a location that allows efficient access.
Additionally, it's important to store relevant metadata for citations or references when the LLM provides responses.

Data chunking: The data in your source needs to be converted to plain text. For example, word documents or PDFs need to be cracked open and converted to text. The text is then chunked into smaller pieces.

Converting the text to vectors: called embeddings. Vectors are numerical representations of concepts converted to number sequences, which make it easy for computers to understand the relationships between those concepts.

### How does a GPT model work?

A GPT model is a type of neural network that uses the transformer architecture to learn from large amounts of text data.
The model has two main components: an encoder and a decoder.
The encoder processes the input text and converts it into a sequence of vectors, called embeddings, that represent the meaning and context of each word.
The decoder generates the output text by predicting the next word in the sequence, based on the embeddings and the previous words.
The model uses a technique called attention to focus on the most relevant parts of the input and output texts, and to capture long-range dependencies and relationships between words.
The model is trained by using a large corpus of texts as both the input and the output, and by minimizing the difference between the predicted and the actual words.
The model can then be fine-tuned or adapted to specific tasks or domains, by using smaller and more specialized datasets.

## Licensing

**ck-llm** is based on open source technologies and released under the [unlicense.org](./LICENSE).

## GitHub Actions

[![Lint Code Base](https://github.com/romitagl/ck-llm/actions/workflows/super-linter.yml/badge.svg)](https://github.com/romitagl/ck-llm/actions/workflows/super-linter.yml)

## Developer Notes

All the pull requests to `master` have to pass the CI check [Super-Linter](https://github.com/github/super-linter).

### Naming Conventions

File & Folder names are separated by `-`, not `_`.

### Super Linter

To setup locally the Super Linter follow [these instructions](https://github.com/github/super-linter/blob/main/docs/run-linter-locally.md).

Run check on the local folder:

```bash
docker run --rm -e RUN_LOCAL=true -v `pwd`:/tmp/lint github/super-linter:latest
```

## Tooling

### Hugging Face

<https://huggingface.co/welcome>

### LLAMA

LLaMA models: <https://github.com/facebookresearch/llama>.

### Ollama

Ollama makes it easy to host any model locally: <https://ollama.ai>.

### roneneldan/TinyStories-33M

Model trained on the TinyStories Dataset: <https://huggingface.co/roneneldan/TinyStories-33M/tree/main>.

### Faiss

A library for efficient similarity search and clustering of dense vectors: <https://github.com/facebookresearch/faiss>.

### LangChain

Building applications with LLMs through composability: <https://github.com/langchain-ai/langchain>.

### Milvus

A cloud-native vector database, storage for next generation AI applications: <https://github.com/milvus-io/milvus>.

### txtai

All-in-one open-source embeddings database for semantic search, LLM orchestration and language model workflows: <https://github.com/neuml/txtai>.
