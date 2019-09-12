# NLP_classification

## 텍스트 분류
데이터 다운로드 : https://www.kaggle.com/c/word2vec-nlp-tutorial/

<데이터 소개>

-	워드 팝콘 : 영화 리뷰 텍스트와 평점에 따른 감정 값으로 구성
-	목표 : 정제, 분석, 알고리즘 모델링

## 데이터 분석 및 전처리
-	데이터를 분류할 수 있는 모델을 학습시킬 것
### 불러오기
-	kaggle 명령어 사용하기 위해서 설치 
~~~
pip install kaggle
~~~

무작정 이렇게 받으면 아래와 같은 OSError가 뜸.
~~~
OSError: Could not find kaggle.json. Make sure it's located in /Users/weehyerin/.kaggle. Or use the environment method.
~~~
kaggle에 회원 가입 되어있고, 그 api token이 컴퓨터에 저장되어 있어야 사용할 수 있나봄..

회원가입 후, kaggle 홈페이지 상단 우측에  Your Profile 을 클릭 후 
 
'Create New API Token' 버튼을 클릭하면 kaggle.json 파일이 다운로드 됨.
<img width="944" alt="스크린샷 2019-09-12 오후 1 48 38" src="https://user-images.githubusercontent.com/37536415/64754918-2d010c00-d564-11e9-923a-71987cfc7799.png">

다운로드 된 kaggle.json 파일은 홈디렉토리에 .kaggle이라는 디렉토리 안에 위치 되면 됨.

이후 권한 설정
~~~
chmod 600 /[홈디렉토리]/.kaggle/kaggle.json
~~~
로 해주면 됨. 


- kaggle에서 다운로드

*약관에 동의를 해야 원하는 파일 다운로드 가능*
<img width="931" alt="스크린샷 2019-09-12 오후 1 55 49" src="https://user-images.githubusercontent.com/37536415/64755125-14452600-d565-11e9-8ae9-a2a3d2627665.png">

그렇지 않으면
~~~
403 forbidden
~~~
이라는 error가 발생하게 됨. 

~~~
kaggle competitions download -c word2vec-nlp-tutorial
~~~
이렇게 명령어 치면 잘 다운로드 됨. 

<img width="564" alt="스크린샷 2019-09-12 오후 1 57 55" src="https://user-images.githubusercontent.com/37536415/64755215-59695800-d565-11e9-9af6-74c68fc8c4a5.png">

- zip file 풀기
다운로드 받은 파일은 unzip 해주어야 함. 
~~~
import zipfile
DATA_IN_PATH = './data_in/'
file_list = ['labeledTrainData.tsv.zip', 'unlabeledTrainData.tsv.zip', 'testData.tsv.zip']

for file in file_list:
    zipRef = zipfile.ZipFile(DATA_IN_PATH + file, 'r')
    zipRef.extractall(DATA_IN_PATH)
    zipRef.close()
~~~

## 데이터 분석

- 데이터 크기
- 데이터의 개수
- 각 리뷰의 문자 길이 분포
- 많이 사용된 단어
- 긍, 부정 데이터의 분포
- 각 리뷰의 단어 개수 분포
- 특수문자 및 대, 소문자 비율

- 데이터 내부 

<img width="452" alt="스크린샷 2019-09-12 오후 2 02 36" src="https://user-images.githubusercontent.com/37536415/64755493-ffb55d80-d565-11e9-8baa-6b16692d5606.png">
>> 각 review에 대한 감정이 sentiment에 1인지, 0인지 나와았음

데이터의 개수 : 25000개

각 리뷰마다 길이 구하기 
~~~
train_length = train_data['review'].apply(len)
~~~

![image](https://user-images.githubusercontent.com/37536415/64759771-17471300-d573-11e9-968c-827ffa53545f.png)

리뷰 길이 분포를 봤을 때, 대부분 6000이하, 혹은 2000이하에 분포되어 있다. 이상치로는 10000자 이상..

- 길이를 기준으로 boxplot 그리기 
![image](https://user-images.githubusercontent.com/37536415/64759868-48bfde80-d573-11e9-8a7e-7ab6a908a26a.png)

> 평균이 2000정도, 4000이상인 이상치 데이터가 많음.


## 단어 살펴보기
![image](https://user-images.githubusercontent.com/37536415/64760044-b1a75680-d573-11e9-905e-2d4ee3614746.png)
워드 클라우로 시각화 했을 때, br이 가장 많이 사용한 것을 알 수 있는데
이는 html tag 중 하나라는 것을 알 수 있다. 따라서 전처리 작업에서 걸러주어야 하는 것을 미리 파악 가능하다. 


긍정의 개수, 부정의 개수는 12500개로 같음

#### 리뷰당 단어의 개수를 확인해보기

![image](https://user-images.githubusercontent.com/37536415/64760191-0ba81c00-d574-11e9-917f-095b05c881e8.png)
리뷰 단어 개수 최대 값: 2470
리뷰 단어 개수 최소 값: 10
리뷰 단어 개수 평균 값: 233.79
리뷰 단어 개수 표준편차: 173.74
리뷰 단어 개수 중간 값: 174.0
리뷰 단어 개수 제 1 사분위: 127.0
리뷰 단어 개수 제 3 사분위: 284.0


> 물음표가있는 질문: 29.55%
> 마침표가 있는 질문: 99.69%
> 첫 글자가 대문자 인 질문: 0.00%
> 대문자가있는 질문: 99.59%
> 숫자가있는 질문: 56.66%

학습에 방해되는 요소 제거 : 대문자 --> 소문자, 특수문자 제거

## 데이터

re, Beautiful soup : 데이터 정제
불용어 제거 : NLTK의 라이브러리 stopwords 모듈 사용
pad_sequences, Tokenizer : 텐서플로우 전처리 모듈
넘파이 : 전처리된 데이터 저장

~~~
import re
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
~~~

###### 첫 번 째 리뷰, 발쵀
~~~
"With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />"
~~~
> html 태그, 특수문자가 포함된 것을 알 수 있음. 
> re, Beautiful Soup를 통해서 제거하기


#### 불용어 삭제하기
불용어 : 문장에서 자주 등장하거나, 전체적인 의미에 영향을 주지 않는 단어
영어에서는 조사, 관사 등의 어휘
노이즈를 줄일 수 있음. 
###### 데이터가 많고 문장 구문에 대한 전체적인 패턴을 모델링 하고자 한다면 불용어를 삭제하지 않아도 됨. 


~~~
LookupError: 
**********************************************************************
Resource stopwords not found.
~~~

NLTK의 stopwords 모듈을 사용하다보면, 위와 같은 에러가 발생할 수 있음.

~~~
import nltk
nltk.download('stopwords')
~~~

라는 명령어를 실행해주면 됨.

~~~
stop_words = set(stopwords.words('english')) # 영어 불용어들의 set을 만든다.

review_text = review_text.lower()
words = review_text.split() # 소문자 변환 후 단어마다 나눠서 단어 리스트로 만든다.
words = [w for w in words if not w in stop_words] # 불용어 제거한 리스트를 만든다
~~~

이렇게 소문자로 변환해서 불용어를 제거해준 후에

~~~
clean_review = ' '.join(words)
~~~
다시 단어 리스트를 문장으로 만들기 위해 합쳐준다.


##### 전처리 함수
~~~
def preprocessing( review, remove_stopwords = False ): 
    # 불용어 제거는 옵션으로 선택 가능하다.
    
    # 1. HTML 태그 제거
    review_text = BeautifulSoup(review, "html5lib").get_text()	

    # 2. 영어가 아닌 특수문자들을 공백(" ")으로 바꾸기
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. 대문자들을 소문자로 바꾸고 공백단위로 텍스트들 나눠서 리스트로 만든다.
    words = review_text.lower().split()

    if remove_stopwords: 
        # 4. 불용어들을 제거
    
        #영어에 관련된 불용어 불러오기
        stops = set(stopwords.words("english"))
        # 불용어가 아닌 단어들로 이루어진 새로운 리스트 생성
        words = [w for w in words if not w in stops]
        # 5. 단어 리스트를 공백을 넣어서 하나의 글로 합친다.	
        clean_review = ' '.join(words)

    else: # 불용어 제거하지 않을 때
        clean_review = ' '.join(words)

    return clean_review
~~~

## 단어를 인덱스화 시키기
전처리한 데이터에서 각 단어를 인섹스로 벡터화, 모델에 따라 입력값의 길이가 동일해야하므로, 일정 길이로 자르고 부족한 부분은 특정값으로 채우는 패딩 과정 진행

1. 텐서플로우의 전처리 모듈 사용하기
~~~
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_reviews)
text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
~~~
정제된 데이터를 tokenizer하고, 인덱스로 구성된 벡터로 변환

~~~
[404, 70, 419, 8815, 506, 2456, 115, 54, 873, 516, 178, 18686, 178, 11242, 165, 78, 14, 662, 2457, 117, 92, 10, 499, 4074, 165, 22, 210, 581, 2333, 1194, 11242,....]
~~~
위와 같이 바뀌게 됨. 

##### 인덱스로 바뀌었는데, 각 인덱스가 어떤 단어인지 알아야할 필요가 있다. 

~~~
word_vocab = tokenizer.word_index
~~~

~~~
{'movie': 1, 'film': 2, 'one': 3, 'like': 4, 'good': 5, 'time': 6, 'even': 7, 'would': 8, 'story': 9, 'really': 10, 'see': 11, 'well': 12, 'much': 13, 'get': 14, 'bad': 15,...}
~~~
hash로 잘 mapping되어 있는 것을 확인할 수 있다. 

전체 단어 개수:  74066

2. 현재 각 데이터는 서로 길이가 다른데, 이 길이를 하나로 통일

이렇게 해야 이후 모델에 바로 적용할 수 있기 때문에 특정 길이를 최대 길이로 정하고 더 긴 데이터의 경우 뒷부분을 자르고 짧은 데이터의 경우에는 0 값으로 패딩하는 작업 진행

~~~
MAX_SEQUENCE_LENGTH = 174 #문장 최대 길이(단어 개수의 통계에서 중간값)

train_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
~~~
pad_sequences : padding 해주는 함수
- 인자 : 데이터, 최대 길이값, 0을 앞에 채울지/뒤에 채울지

##### 위의 결과로 데이터가 모두 174라는 길이를 가지게 됨. 
한 개의 리뷰가 한 개의 174라는 길이를 가지는 벡터임. 

## 마지막으로 위의 전처리한 결과들을 각각 저장하기
- 정제된 텍스트 데이터: 특수문자, html tag 등을 삭제한 것
- 벡터화한 데이터 : padding 처리한 후 데이터
- 정답라벨 : 긍정, 부정
- 데이터 정보(단어 사전, 전체 단어 개수)


# 모델링

> 위의 전처리된 데이터를 가지고 직접 모델에 적용하고 
> 주어진 텍스트에 대해 감정이 긍정인지 부정인지 예측할 수 있는 모델 만들기

## 1. 회귀 모델

로지스틱 회귀 모델 : 이항 분류를 하기 위해 사용, 분류 모델에서 사용할 수 있는 간단한 모델
- 선형 결합을 대로 예측하면 됨

### 선형 회귀 모델
- 종속변수와 독립변수 간의 상관관계를 모델링하는 방법

### 로지스틱 회귀 모델
로지스틱 함수를 적용해 0 ~ 1 사이의 값을 갖게 해서 확률로 표현
> 결과가 1에 가까우면 1, 0에 가까우면 0이라고 예측

### TF-IDF 사용하기
위에서 정제한 텍스트 데이터(불용어, html tag 등 삭제)한 것을 벡터화 

~~~
vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3), max_features=5000) 

X = vectorizer.fit_transform(reviews)
y = np.array(sentiments)
~~~

- min_df : 설정한 값보다 특정 토큰의 df 값이 더 적게 나오면 벡터화 과정에서 제거한다.
- analyzer : 분석하기 위한 기준 단위(word : 단어 하나를 단위로 하는 것, char : 문자 하나를 단위로)
- sublinear_tf : 문서의 단어 빈도 수에 대한 smoothing 여부를 설정
- ngram_range : 빈도의 기본 단위를 어느 범위의 n_gram으로 설정할 것인지 
- max_featuer : 각 벡터의 최대 길이, 특징의 길이를 설정


### Word2Vec 사용하기
word2vec의 경우, 단어로 표현된 리스트를 넣어야 함. 따라서 정제한 텍스트 데이터를 가지고 와서 잘라주기

~~~
sentences = []
for review in reviews:
    sentences.append(review.split())
~~~

word2vec 모델의 하이퍼파라미터 설정
~~~
num_features = 300    # 워드 벡터 특징 값 수, 각 단어에 대해 임베딩된 벡터의 차원 설정
min_word_count = 40   # 단어에 대한 최소 빈도 수, 모델에 의미 있는 단어를 가지고 학습하기 위해 적은 빈도 수의 단어들은 학습하지 않는다.
num_workers = 4       # 프로세스 개수
context = 10          # 컨텍스트 윈도우 크기
downsampling = 1e-3   # 다운 샘플링 비율, 학습을 수행할 때 빠른 학습을 위해 정답 단어 라벨에 대한 다운 샘플링 비율을 지정
~~~

~~~
pip install --upgrade pip
~~~
pip ungrade 이후, 

~~~
!pip install gensim
~~~
gensim 라이브러리 설치 : gensim의 word2vec 모듈을 불러오기 위해서

~~~
from gensim.models import word2vec

model = word2vec.Word2Vec(sentences, workers=num_workers, \
           size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
~~~
word2vec으로 학습

##### word2vec 모델의 경우, 모델을 저장해 두면 이후에 다시 사용할 수 있다. 
~~~
model_name = "300features_40minwords_10context"
model.save(model_name)
~~~

~~~
Word2Vec.load() 를 통해서 모델을 다시 사용 가능
~~~

##### word2vec을 한 이후에, 각 텍스트들을 같은 크기의 벡터로 바꿔주기

## 2. 랜덤포레스트 모델

### CountVectorizer를 활용한 벡터화
- 텍스트 데이터에서 횟수를 기준으로 특징을 추출하는 방법
- 어떤 단위의 횟수를 셀 것인지는 선택 사항
- 단위는 단어가 될 수도 있고, 문자 하나나가 될 수도 있다. 
- 보통은 텍스트에서 단어를 기준으로 횟수를 측정, 문장을 입력으로 받아 단어의 횟수를 측정한 뒤 벡터로 만든다. 

> 모델의 입력값은 전처리한 텍스트 데이터

~~~
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000) 

train_data_features = vectorizer.fit_transform(reviews)
~~~
- 분석 단위 : 하나의 단어(word) / 벡터의 최대 길이 : 5000

##### 모델 학습
~~~
from sklearn.ensemble import RandomForestClassifier


# 랜덤 포레스트 분류기에  100개 의사 결정 트리를 사용한다.
forest = RandomForestClassifier(n_estimators = 100) 

# 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작한다.
forest.fit( train_input, train_label )
~~~

## 3. 순환 신경망 분류 모델
순환 신경망은 언어 모델에서 많이 쓰이는 딥러닝 모델 중 하나. 
주로 순서가 있는 데이터, 즉 문장 데이터를 입력해서 문장 흐름에서 패턴을 찾아 분류하게 한다. 

>> 텍스트 자체를 입력해서 문장에 대한 특징 정보를 추출

![image](https://user-images.githubusercontent.com/37536415/64771124-6ac45b80-d589-11e9-9000-1d5a25df92fe.png)

![image](https://user-images.githubusercontent.com/37536415/64765260-c50bef00-d57e-11e9-8e34-d88f6c90cba2.png)

위의 모델은 한 단어에 대한 정보를 입력하면 이 단어 다음에 나올 단어를 맞추는 모델
> 아버지라는 단어 정보를 모델에 입력해서 그 다음에 나올 단어를 예측하고, 
> 그 다음에 가방에라는 정보가 입력되면, 앞서 입력한 아버지라는 정보를 입력해서 처리된 정보와 함께 활용해서 다음 단어를 예측
> 현재정보를 input sate 이전 정보를 hiddne state --> 순환 신경망은 이 두 상태 정보를 활용해 순서가 있는 데이터에 대한 예측 모델링을 가능하게 함.

![그림1](https://user-images.githubusercontent.com/37536415/64771047-4a949c80-d589-11e9-88de-a9ea2958ce66.png)
입력 문장을 순차적으로 입력만 하고 마지막으로 입력한 시점에 출력 정보를 뽑아 영화 평점을 예측

전처리한 데이터 그대로 사용

###### 하이퍼 파라미터
~~~
WORD_EMBEDDING_DIM = 100
HIDDEN_STATE_DIM = 150
DENSE_FEATURE_DIM = 150

learning_rate = 0.001
~~~

###### 임베딩 층
~~~
embedding_layer = tf.keras.layers.Embedding(
                    VOCAB_SIZE,
                    WORD_EMBEDDING_DIM)(features['x'])
~~~

모델에서 배치 데이터를 받게 되면, 단어 인덱스로 구성된 시퀀스 형태로 입력이 들어옴
모델에 들어온 입력 데이터는 임베딩 층을 거침

###### rnn
~~~
 rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [HIDDEN_STATE_DIM, HIDDEN_STATE_DIM]]
 multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
~~~
- 문장의 의미 벡터 만듦

> 순환신경망을 구현하기 위해 RNNCell 이용
>> LSTM으로 순환 신경망을 구성하기 위해서는 (tf.nn.rnn_cell.LSTMCell)로 사용

노드를 다 구성하면 MultiRNN으로 묶어주어야 한다. --> MultiRNNCell(rnn_layers)

~~~
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=embedding_layer,
                                       dtype=tf.float32)
~~~
RNNCell(RNN의 노드 하나)에서는 시퀀스 한 스텝에 대한 연산만 가능
> 여러 스텝에 대한 연산을 하기 위해 dynamic_rnn 사용
>> 인자 : rnn 객체인 MultiRNNCell / 입력값 / dtype으로 출력값의 type 지정

~~~
hidden_layer = tf.keras.layers.Dense(DENSE_FEATURE_DIM, activation=tf.nn.tanh)(outputs[:,-1,:])
~~~
위의 dynamic_rnn의 과정에서 나온 출력값을 Dense에 적용

~~~
logits = tf.keras.layers.Dense(1)(hidden_layer)
~~~
마지막으로 긍/부정을 판단하도록 출력값을 하나로 만듦. --> 입력벡터에 대한 차원수를 Dense를 통해 바꿈
~~~
test_id = np.load(open(DATA_IN_PATH + TEST_ID_DATA, 'rb'))
~~~

위의 부분에서 에러가 나면 (allow_pickle=False라는 에러) 아래처럼 처리
~~~
test_id = np.load(open(DATA_IN_PATH + TEST_ID_DATA, 'rb'), allow_pickle=True)
~~~

## 4. CNN
> 합성곱 신경망 : 전통적인 신경망 앞에 여러 계층의 합성곱 계층을 쌓은 모델, 입력 받은 이미지에 대한 가장 좋은 특징을 만들어 내도록 학습하고, 추출된 특징을 활용해 이미지를 분류하는 방식
>> 일반적으로 이미지에서 강아지, 고양이, 돼지 등과 같은 특정 라벨을 붙여 데이터 셋을 만들고, 모델이 학습을 하면서 각 특징값을 추출해서 특징을 배우고, 가장 가까운 라벨을 예측
![image](https://user-images.githubusercontent.com/37536415/64772715-0c4cac80-d58c-11e9-900f-1d6d0404ba83.png)

### CNN에서 텍스트를 어떻게 사용할까?
- RNN이 단어의 입력 순서를 중요하게 반영한다면, CNN은 문장의 지역 정보를 보존하면서 각 문장 성분의 등장 정보를 학습에 반영하는 구조
> 학습할 때, 각 필터 크기를 조절하면서 언어의 특징 값(언어의 벡터)을 추출하게 되는데 n-gram 방식과 유사 







