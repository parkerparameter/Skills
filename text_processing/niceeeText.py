import re
import json
import pandas as pd
import pickle
import random
import numpy as np
import operator
import gensim.models
from gensim.test.utils import datapath
from gensim import utils
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
    strip_numeric, remove_stopwords, strip_short
from nltk.tokenize import sent_tokenize
from scipy import spatial

# Greeting Inputs
GREETING_INPUTS = ["hi", "hello", "hola", "greetings", "wassup", "hey"]

# Greeting responses back to the user
GREETING_RESPONSES = ["hi", "hey", "what's good", "hello", "hey there"]


def aggregate_embeddings(list_of_embeddings):
    """
    takes simple average of embeddings to produce "Average" context of the words
    :param list_of_embeddings: list of numpy array of same dimension
    :return: numpy array
    """

    return np.mean(list_of_embeddings, axis=0)


def is_greeting(sentence):
    if any([True if word.lower() in GREETING_INPUTS else False for word in sentence.split()]):
        return True
    return False


class niceeeText:

    def __init__(self):
        self.w2v = gensim.models.Word2Vec.load('untuned_twit2vec').wv
        with open('answer_embeddings.pickle', 'rb') as file:
            self.key = pickle.load(file)

    @staticmethod
    def remove_emojis(text_array):
        """
        remove emoji unicode characters from  text array
        :param text_array: numpy array of text
        :return: numpy array of text, w/o emojis
        """
        # emoji regular expression #

        emoji_pattern = re.compile(
            "["
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"
            "]+"
        )

        # assert(isinstance(text,np.ndarray)) # always feed a numpy array

        return np.array([re.sub(emoji_pattern, "", str(string)) for string in text_array])

    @staticmethod
    def remove_handles(text_array):
        """
        remove twitter handles from an array of txt
        :param text_array: numpy array of text
        :return: text array without handles
        """

        handle_pattern = re.compile(r'([@])\w+')

        return np.array([re.sub(handle_pattern, "", str(string)) for string in text_array])

    @staticmethod
    def replace_hashtags(text_array,
                         repchar=' '):
        """
        replace hashtags in array of text with repchar
        :param text_array: numpy array of text
        :param repchar: replacement character, must be a string
        :return: numpy array of text with replaced hashtags
        """

        assert isinstance(repchar, str)

        hashtag_pattern = re.compile(r'([#])\w+')

        return np.array([re.sub(hashtag_pattern, repchar, str(string)) for string in text_array])

    def total_twitter_scrub(self, text_array):
        """
        wrapper to call each of the three above at once
        :param text_array: array of text
        :return: array of text
        """

        return self.replace_hashtags(self.remove_handles(self.remove_emojis(text_array)))

    @staticmethod
    def process_text(document):
        """
        gensim text preprocessing wrapper minus lemmatization bc of acronyms
        :param document: document of text to be preprocessed
        :return: clean tokens of the document
        """
        return preprocess_string(document,
                                 filters=[strip_tags, strip_punctuation,
                                          strip_multiple_whitespaces,
                                          strip_numeric, remove_stopwords,
                                          strip_short]
                                 )

    def clean_corpus(self, corpus, flatten=True):
        """
        clean corpus of text and tokenize
        :param flatten: boolean; should return flat list of words or list of lists [sentences]
        :param corpus: raw string of corpus
        :return: list of tokenized, clean sentences from corpus
        """
        if flatten:
            t = [self.process_text(sent) for sent in list(sent_tokenize(corpus))]
            return [item for sublist in t for item in sublist]
        return [self.process_text(sent) for sent in list(sent_tokenize(corpus))]

    def embed_word(self, word):
        """
        embed word
        :param word: string, word to embed
        :return: numpy array; word embedding according to self.w2v
        """

        try:
            return self.w2v[str(word)]
        except KeyError:
            return [np.nan]  # gotta do something else here

    def embed_corpus(self, list_of_words):
        """
        call self.word for word in list of words
        :param list_of_words: list of cleaned words to be embeded
        :return: list of numpy array: w2v embedding
        """
        return [self.embed_word(word) for word in list_of_words if not all(np.isnan(self.embed_word(word)))]

    def clean_then_embed(self, corpus):
        """
        wrapper to call self.embed_corpus and self.clea_corpus
        :param corpus:
        :return: list of numpy array; w2v embedding
        """

        return self.embed_corpus(self.clean_corpus(corpus=corpus))

    def fetch_most_similar_answer(self, text, unclean=True, threshold=.25):
        """
        fetch most similar answer for text query
        :param text: input text
        :param unclean: boolean, please just pass raw text values
        :return: string of most similar answer
        """
        if is_greeting(text):
            return random.choice(GREETING_RESPONSES)
        if unclean:
            inp_embedded = aggregate_embeddings(self.clean_then_embed(text)).reshape(-1, 1)
            comparison_dict = {k: spatial.distance.cosine(v.reshape(-1, 1), inp_embedded) for k, v in
                               self.key.items()}
            # dont get lost, just calculating the cosine similarity between this and every answer in the corpus #

            if max(comparison_dict.items(), key=operator.itemgetter(1))[1] < threshold:
                return "I apologize, I don't understand"
            return max(comparison_dict.items(), key=operator.itemgetter(1))[0]
