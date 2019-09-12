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



- kaggle에서 다운로드
~~~

~~~
