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


















