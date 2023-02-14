import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class Dataset:

    def __init__(self, args):

        self.random_state = args.seed
        self.dataset_name = args.dataset_name
        self.intent = self._load_dataset(args)

        # tokenize
        self._get_tokenizer(args)

        # featurize
        # BERT의 [CLS] 토큰을 featurizer로 활용하는 경우
        if args.featurizer_model == 'BERT':
            tokenizer, model = self._get_featurizer(args)
            utterance_inputs = tokenizer(self.intent['utterance'].to_list(), return_tensors='pt', padding=True, truncation=True)

            utterance_feat_ = model(**utterance_inputs).pooler_output

            self.utterance_feat = utterance_feat_.detach().cpu()

        # Count 기반(CountVectorizer 또는 TF-IDF)의 featurizer를 활용하는 경우
        else:
            self.featurizer = self._get_featurizer(args)
            self.utterance_featurizer = self.featurizer()

            utterance_tmp = [' '.join(self.tokenizer.tokenize(sentence)) for sentence in self.intent['utterance']]
            utterance_feat_ = self.utterance_featurizer.fit_transform(utterance_tmp)
            self.utterance_feat = utterance_feat_.toarray()

        self.test_size = args.test_size
        self.categ2label = ''


    def get_dataset(self):

        # X = utterance_feat_.toarray()
        X = self.utterance_feat

        # 레이블=> categorical
        label2categ = {v: i for i, v in enumerate(set(self.intent['intent']))}

        self.n_labels=len(label2categ)
        self.categ2label = {label2categ[key]: key for key in label2categ}
        y = [label2categ[key] for key in self.intent['intent']]

        return self._split_data(X, y, indices=np.arange(len(X)), test_size=self.test_size, random_state=self.random_state), self.n_labels, self.categ2label

    def _load_dataset(self, args):
        # To Do
        # 전처리 작업까지 구현 필요
        intent = pd.read_csv(args.dataset_name)
        intent = intent[["utterance", "intent"]]
        return intent.fillna("")

    def _get_tokenizer(self, args):
        # To Do
        # KoNLPy, Mecab 등 다양한 토크나이저 구현 필요
        self.tokenizer = AutoTokenizer.from_pretrained(args.BERTtokenizer_model)

    def _get_featurizer(self, args):
        if args.featurizer_model == 'TfidfVectorizer':
            return TfidfVectorizer
        elif args.featurizer_model == 'CountVectorizer':
            return CountVectorizer
        elif args.featurizer_model == 'BERT':
            tokenizer = AutoTokenizer.from_pretrained(args.BERTtokenizer_model)
            model = AutoModel.from_pretrained(args.BERTtokenizer_model)
            return tokenizer, model


    def _split_data(self, X, y, indices, test_size, random_state):
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, indices, test_size=test_size, random_state=random_state)