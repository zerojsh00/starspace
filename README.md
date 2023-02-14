# StarSpace를 활용한 임베딩 분류기

본 레포지터리에서는 few-show learning의 일환으로 요긴하게 활용할 수 있는 [StarSpace](https://arxiv.org/pdf/1709.03856.pdf) 분류기 학습을 체험할 수 있습니다.

2017년 facebook AI에서 제안된 StarSpace는 텍스트 분류를 포함하여 다양한 도메인에서 활용될 수 있는 신경망 임베딩 모델(a general purpose neural embedding model)입니다. 
특히, 적은 양의 데이터로도 벡터를 탁월하게 분류할 수 있어, 챗봇의 의도분류기로 활용될 수 있습니다.

## 기본 메커니즘

### 1) 문장을 토크나이즈 함
   - 단순한 공백을 기준으로 split 하는 방식 `To Do`
   - KoNLPy, Mecab 등의 형태소 분석기를 활용하여 토크나이즈 하는 방식 `To Do`
   - BERT의 subword tokenizer를 활용하는 방식 `Default Method`
     - `--BERTtokenizer_model='klue/bert-base'`


### 2) 토큰 시퀀스를 벡터화 함
   - CountVectorizer를 활용한 방식
     - 단순한 방법이지만, 특정 단어들의 출현이 중요한 closed domain에서 효과적일 것으로 기대됨
     - `--featurizer_model='CountVectorizer'`
   - TF-IDF를 활용한 방식
     - 단순한 방법이지만, 특정 단어들의 출현이 중요한 closed domain에서 효과적일 것으로 기대됨
     - CountVectorizer보다 문서 내 단어들의 출현 정보를 잘 표현하여 효과적임
     - `--featurizer_model='TfidfVectorizer'`
   - BERT의 [CLS] 토큰 벡터를 활용한 방식
     - 대량의 일반 도메인 코퍼스를 통해서 사전학습한 모델을 활용함으로써, 다소 무겁지만 open domain에서 효과적일 것으로 기대됨
     - `--featurizer_model='BERT'`
     - `--BERTtokenizer_model='klue/bert-base'`


### 3) StarSpace 임베딩 분류기를 통해 분류함
   - 특징 벡터들에 대해 linear layer를 통과시켜 벡터 공간 상에 projection 함
   - `nn.Embedding`으로 구현되어 있는 레이블 임베딩들에 대해 linear layer를 통과시켜 벡터 공간 상에 projection 함
   - 특징 벡터와 레이블 임베딩들을 내적하여 similarity score를 구함
     - 특징 벡터와 '정답' 레이블 벡터와는 내적의 결과가 클 것으로 기대함
     - 특징 벡터와 '오답' 레이블 벡터와는 내적의 결과가 작을 것으로 기대함
   - similarity score의 softmax 값과 실제 정답 레이블의 차이에 대해 cross entropy loss를 구함
   - 즉, 특징 벡터가 '정답' 레이블(positive sample)과는 가깝게, '오답' 레이블(negative samples)과는 멀게 임베딩 되도록 함


## 학습 방법

```python
# default setting : TF-IDF 벡터 기반의 StarSpace 분류기
python train.py 

# CountVectorizer 벡터 기반의 StarSpace 분류기
python train.py --featurizer_model='CountVectorizer'

# BERT의 [CLS] 토큰 벡터 기반의 StarSpace 분류기
# default 사전학습 BERT는 "klue/bert-base"임
python train.py --featurizer_model='BERT' --BERTtokenizer_model='klue/bert-base'
```

## To Do

- inference 코드의 작성
- logger 및 wandb 사용
- 전처리 코드의 통합
- 배치 단위의 학습 지원
  - 현재는 극소량의 데이터를 간주하여 별도의 미니배치 학습을 구현하지 않음
- 다양한 tokenizer 및 featurizer 지원
- 기타 비효율적인 코드 개선