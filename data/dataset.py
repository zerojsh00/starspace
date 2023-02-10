import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
class Dataset:
    def __init__(self, args):
        if args.vectorizer_name == 'TfidfVectorizer':
            self.utterance_vectorizer = TfidfVectorizer()
        elif args.vectorizer_name == 'CountVectorizer':
            self.utterance_vectorizer = CountVectorizer()
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
        self.dataset_name = args.dataset_name
        self.test_size = args.test_size
        self.random_state = args.seed
        self.categ2label = ''
    def load_dataset(self):
        # To Do
        # 전처리 작업까지 구현 필요
        intent = pd.read_csv(self.dataset_name)
        intent = intent[["utterance", "intent"]]
        intent = intent.fillna("")
        intent['utterance_'] = [' '.join(self.tokenizer.tokenize(sentence)) for sentence in intent['utterance']]
        utterance_feat_ = self.utterance_vectorizer.fit_transform(intent['utterance'])
        X = utterance_feat_.toarray()

        # 레이블 본부 => categorical
        label2categ = {v: i for i, v in enumerate(set(intent['intent']))}

        self.n_labels=len(label2categ)
        self.categ2label = {label2categ[key]: key for key in label2categ}
        y = [label2categ[key] for key in intent['intent']]
        return self._split_data(X, y, indices=np.arange(len(X)), test_size=self.test_size, random_state=self.random_state), self.n_labels, self.categ2label
    def _split_data(self, X, y, indices, test_size, random_state):
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, indices, test_size=test_size, random_state=random_state)