from collections import OrderedDict, Counter, defaultdict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
import heapq
import itertools
import math

class BoW(TransformerMixin):
    """
    Bag of words transformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurrences (the highest first)
        # store most frequent tokens in self.bow

        freq_dict = defaultdict(int)
        for sentence in X:
            for word in sentence.split():
                freq_dict[word] += 1
        
        self.bow = sorted(freq_dict, key=freq_dict.get, reverse=True)[:self.k]

        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = {key: 0 for key in self.bow}

        text_dict_freq = {key: 0 for key in text.split()}

        for t in text.split():
            if t in self.bow:
                text_dict_freq[t] += 1

        final_result = {key : text_dict_freq.get(key, val) \
                        for key, val in result.items()}

        final_list = [v for k, v in final_result.items()]

        return np.array(final_list, "float32")


    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf transformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        if self.k is None:
            self.k = len(set(' '.join(X).split()))

        N = len(X)
        freq_dict = defaultdict(int)
        self.idf = defaultdict(int)

        for sentence in X:
            for word in sentence.split():
                freq_dict[word] += 1
                self.idf[word] = np.log10(N / (freq_dict.get(word)))

        self.idf = dict(sorted(self.idf.items(), key=lambda x:x[1],reverse=True))
        self.idf = dict(itertools.islice(self.idf.items(), self.k))
        
        return self


    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        result = {key: 0 for key in self.idf}
                
        n = len(text.split())
        text_dict_freq = defaultdict(int)
        tf = defaultdict(int)
        for t in text.split():
            if t in result.keys():
                text_dict_freq[t] += 1
                tf[t] = text_dict_freq.get(t) / n       
        
        for k in tf:
            if k in result:
                result[k] = tf[k] * self.idf[k]

        final_list = [v for k, v in result.items()]

        if self.normalize is True:            
            final_list = final_list / (np.linalg.norm(final_list) + 1e-5) 
     
        return np.array(final_list, "float32")


    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])



