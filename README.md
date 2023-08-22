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

### How does a GPT model work?

A GPT model is a type of neural network that uses the transformer architecture to learn from large amounts of text data.
The model has two main components: an encoder and a decoder.
The encoder processes the input text and converts it into a sequence of vectors, called embeddings, that represent the meaning and context of each word.
The decoder generates the output text by predicting the next word in the sequence, based on the embeddings and the previous words.
The model uses a technique called attention to focus on the most relevant parts of the input and output texts, and to capture long-range dependencies and relationships between words.
The model is trained by using a large corpus of texts as both the input and the output, and by minimizing the difference between the predicted and the actual words.
The model can then be fine-tuned or adapted to specific tasks or domains, by using smaller and more specialized datasets.

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

<https://github.com/facebookresearch/llama>

### LangChain

<https://github.com/langchain-ai/langchain>

### Milvus

<https://github.com/milvus-io/milvus>
