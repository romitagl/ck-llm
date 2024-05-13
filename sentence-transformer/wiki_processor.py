import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")


# Specify the directory containing the .md files
def process_wiki_dir(wiki_dir="kgraph.wiki"):
    # Initialize an empty list to store the preprocessed data
    preprocessed_data = []

    # Iterate over all .md files in the wiki_dir directory
    for filename in os.listdir(wiki_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(wiki_dir, filename)
            with open(file_path, "r") as file:
                wiki_data = file.read()

            # Tokenize the text into sentencesf
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

    return preprocessed_data

