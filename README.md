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











