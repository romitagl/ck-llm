# ck-llm

Custom Knowledge LLM - Use a base LLM and infuse structured knowledge

## Quickstart

```bash
pip install huggingface_hub
huggingface-cli login
```

### Llama-2-7b

```bash
cd llama-2-7b-hf
python -m pip install -r requirements.txt 
python text-completion.py
```

Run in a `C` program: <https://github.com/karpathy/llama2.c>/.

### Notes

LlamaTokenizer requires the SentencePiece library. Checkout the instructions on the
installation page of its repo: <https://github.com/google/sentencepiece#installation>.

On Mac, install the sentencepiece library with: `brew install sentencepiece`.

## Description

Traditionally, a base model is trained with point-in-time data to ensure its effectiveness in performing specific tasks and adapting to the desired domain. However, sometimes you need to work with newer or more current data.
Two approaches can supplement the base model:

- fine-tuning or further training of the base model with new data
- RAG that uses prompt engineering to supplement or guide the model in real time.

Fine-tuning is suitable for continuous domain adaptation, enabling significant improvements in model quality but often incurring higher costs.
Conversely, RAG offers an alternative approach, allowing the use of the same model as a reasoning engine over new data provided in a prompt.
This technique enables in-context learning without the need for expensive fine-tuning, empowering businesses to use LLMs more efficiently.

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

#### Example

Let's see how to finetune a language model to generate character backstories using HuggingFace Trainer with wandb integration. We'll use a tiny language model (TinyStories-33M) due to resource constraints: [./tinystories-33m/05_train_llm_starter.ipynb](tinystories-33m/05_train_llm_starter.ipynb)

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

#### LLAMA

LLaMA models: <https://github.com/facebookresearch/llama>.

#### roneneldan/TinyStories-33M

Model trained on the TinyStories Dataset: <https://huggingface.co/roneneldan/TinyStories-33M/tree/main>.

### Faiss

A library for efficient similarity search and clustering of dense vectors: <https://github.com/facebookresearch/faiss>.

### LangChain

Building applications with LLMs through composability: <https://github.com/langchain-ai/langchain>.

### Milvus

A cloud-native vector database, storage for next generation AI applications: <https://github.com/milvus-io/milvus>.

### txtai

All-in-one open-source embeddings database for semantic search, LLM orchestration and language model workflows: <https://github.com/neuml/txtai>.
