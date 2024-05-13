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

from retrieval_based_qa import answer_question

# Get user input for the question
user_question = input("Enter your question: ")
if user_question == "":
    user_question = "What are some best practices for writing Dockerfiles?"
answers = answer_question(user_question)
print(f"User Question: {user_question}\n Answers:")
for answer in answers:
    print(answer)
