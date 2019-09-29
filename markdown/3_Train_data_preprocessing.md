
# 라이브러리 불러오기


```python
import pandas as pd 
import numpy as np

# 시각화 준비
import matplotlib
import matplotlib.pyplot as plt

# Jupyter Notebook 내부에 그래프를 출력하도록 설정
%matplotlib inline

# Seaborn은 Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# 한글 깨지는거 방지하기
from matplotlib import font_manager, rc

import platform
if platform.system() == 'Windows':
# 윈도우인 경우
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
else:    
# Mac 인 경우
    rc('font', family='AppleGothic')
matplotlib.rcParams['axes.unicode_minus'] = False   
#그래프에서 마이너스 기호가 표시되도록 하는 설정

# datetime 객체 추가
from datetime import datetime, timedelta
```




```python

```

# train_schedule 불러와서 train data 정제 -> train


```python
# pred에서 한 작업을 똑같이 진행

train = pd.read_csv("../input/AFSNT.CSV", engine = "python", encoding ='CP949')
print('Dataframe dimensions:', train.shape)
print(train.head())

train.columns = ['년','월','일', '요일', '공항', '상대공항', '항공사', '항공편', '식별번호', '출도착',
             '부정기편','계획', '실제', '지연', '지연사유', '결항', '결항사유']
recollist = ['년','월','일', '요일', '공항', '출도착', '상대공항', '항공사', '항공편', '식별번호', 
              '계획', '실제', '지연', '지연사유', '결항', '결항사유', '부정기편']
train = train[recollist]


transairport = {'ARP1':'김포', 'ARP2':'김해', 'ARP3': '제주', 'ARP4':'대구', 'ARP5': '울산',
           'ARP6':'청주', 'ARP7':'무안', 'ARP8':'광주', 'ARP9':'여수', 'ARP10':'양양',
           'ARP11':'포항', 'ARP12':'사천', 'ARP13':'군산', 'ARP14':'원주', 'ARP15':'인천'}
train['공항'] = train['공항'].apply(lambda x: transairport[x])
train['상대공항'] = train['상대공항'].apply(lambda x: transairport[x])
train["출도착"] = train["출도착"].apply(lambda x: "출발" if x=="D" else x)
train["출도착"] = train["출도착"].apply(lambda x: "도착" if x=="A" else x)
#df = df.iloc[df['항공사']!= ['C', 'D', 'E', 'G', 'K'],]
transairline = {'A':'아시아나', 'B':'에어부산', 'C': '전일본항공ANA', 'D':'에어서울', 'E': '불명',
           'F':'이스타', 'G':'일본항공', 'H':'제주항공', 'I':'진에어', 'J':'대한항공',
           'K':'타이완', 'L':'티웨이', 'M':'신규'}
train['항공사'] = train['항공사'].apply(lambda x: transairline[x])
# train.to_pickle("train.pickle")
train.head(5)
```

    Dataframe dimensions: (987709, 17)
       SDT_YY  SDT_MM  SDT_DD SDT_DY   ARP   ODP FLO    FLT       REG AOD IRR  \
    0    2017       1       1      일  ARP3  ARP6   J  J1955  SEw3NzE4   D   N   
    1    2017       1       1      일  ARP3  ARP6   J  J1954  SEw3NzE4   A   N   
    2    2017       1       1      일  ARP3  ARP6   J  J1956  SEw3NzE4   A   N   
    3    2017       1       1      일  ARP3  ARP6   J  J1957  SEw3NzE4   D   N   
    4    2017       1       1      일  ARP3  ARP6   J  J1958  SEw3NzE4   A   N   
    
         STT    ATT DLY  DRR CNL  CNR  
    0  10:05  10:32   N  NaN   N  NaN  
    1   9:30   9:31   N  NaN   N  NaN  
    2  12:45  13:03   N  NaN   N  NaN  
    3  13:25  14:09   Y  C02   N  NaN  
    4  16:10  16:31   N  NaN   N  NaN  





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>년</th>
      <th>월</th>
      <th>일</th>
      <th>요일</th>
      <th>공항</th>
      <th>출도착</th>
      <th>상대공항</th>
      <th>항공사</th>
      <th>항공편</th>
      <th>식별번호</th>
      <th>계획</th>
      <th>실제</th>
      <th>지연</th>
      <th>지연사유</th>
      <th>결항</th>
      <th>결항사유</th>
      <th>부정기편</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>10:05</td>
      <td>10:32</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1954</td>
      <td>SEw3NzE4</td>
      <td>9:30</td>
      <td>9:31</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1956</td>
      <td>SEw3NzE4</td>
      <td>12:45</td>
      <td>13:03</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>13:25</td>
      <td>14:09</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1958</td>
      <td>SEw3NzE4</td>
      <td>16:10</td>
      <td>16:31</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python
train.shape
```




    (987709, 17)



# train 결항 제거


```python

# def del_cancle(row):
#     if row['결항'] == 'Y':
#         print(row)
#         del row
    
# train.apply(del_cancle, axis = 1)

train = train[train.결항 == "N"]
train.shape
```




    (979450, 17)




```python
# 계획, 실제 datetime으로 변환
# 시간값만 뽑은 칼럼 만들고 +1일 할 것 찾아서 치환
# 출발일 경우 x_차이 col에 대입
# 도착일 경우 y_차이 col에 대입
# x_차이 > 30 x_차이 < -30 인데 N라고 한것들 제외
# y_차이 > 30 y_ckdl < -30인데 N라고 한것들 제외
# x_차이 <= 30 and x_차이 >= -30 인데 Y라고 한것들 제외
# y_차이 <= 30 and y_차이 >= -30 인데 Y라고 한것들 제외

# x_차이, y_차이, time_exp, time_act 칼럼 지우기
# 계획, 실제 time만 가져와서 대입
```


```python
train['계획'] = pd.to_datetime(train['계획'])
train['실제'] = pd.to_datetime(train['실제'])
print(train.head())
```

          년  월  일 요일  공항 출도착 상대공항   항공사    항공편      식별번호                  계획  \
    0  2017  1  1  일  제주  출발   청주  대한항공  J1955  SEw3NzE4 2019-09-07 10:05:00   
    1  2017  1  1  일  제주  도착   청주  대한항공  J1954  SEw3NzE4 2019-09-07 09:30:00   
    2  2017  1  1  일  제주  도착   청주  대한항공  J1956  SEw3NzE4 2019-09-07 12:45:00   
    3  2017  1  1  일  제주  출발   청주  대한항공  J1957  SEw3NzE4 2019-09-07 13:25:00   
    4  2017  1  1  일  제주  도착   청주  대한항공  J1958  SEw3NzE4 2019-09-07 16:10:00   
    
                       실제 지연 지연사유 결항 결항사유 부정기편  
    0 2019-09-07 10:32:00  N  NaN  N  NaN    N  
    1 2019-09-07 09:31:00  N  NaN  N  NaN    N  
    2 2019-09-07 13:03:00  N  NaN  N  NaN    N  
    3 2019-09-07 14:09:00  Y  C02  N  NaN    N  
    4 2019-09-07 16:31:00  N  NaN  N  NaN    N  



```python
train['time_exp'] = train['계획'].apply(lambda x : x.time().hour)
train['time_act'] = train['실제'].apply(lambda x : x.time().hour)

def nextday_time_mapping(row):
  if (row['time_exp'] >= 20) and (row['time_act'] >=0) and (row['time_act'] <= 5):
    print(row)
    return row['실제'] + timedelta(days = +1)
  else:
    return row['실제']

train.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>년</th>
      <th>월</th>
      <th>일</th>
      <th>요일</th>
      <th>공항</th>
      <th>출도착</th>
      <th>상대공항</th>
      <th>항공사</th>
      <th>항공편</th>
      <th>식별번호</th>
      <th>계획</th>
      <th>실제</th>
      <th>지연</th>
      <th>지연사유</th>
      <th>결항</th>
      <th>결항사유</th>
      <th>부정기편</th>
      <th>time_exp</th>
      <th>time_act</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 10:05:00</td>
      <td>2019-09-07 10:32:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1954</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 09:30:00</td>
      <td>2019-09-07 09:31:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1956</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 12:45:00</td>
      <td>2019-09-07 13:03:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 13:25:00</td>
      <td>2019-09-07 14:09:00</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1958</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 16:10:00</td>
      <td>2019-09-07 16:31:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>16</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
today_day = pd.to_datetime('today').day
train['실제'] = train.apply(nextday_time_mapping, axis = 1)

```

    년                          2017
    월                            10
    일                             2
    요일                            월
    공항                           제주
    출도착                          도착
    상대공항                         청주
    항공사                         진에어
    항공편                       I1559
    식별번호                   SEw3NTYz
    계획          2019-09-07 20:25:00
    실제          2019-09-07 00:05:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     20
    time_act                      0
    Name: 354335, dtype: object
    년                          2018
    월                             2
    일                             5
    요일                            월
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        대한항공
    항공편                       J1707
    식별번호                   SEw3NzY0
    계획          2019-09-07 23:05:00
    실제          2019-09-07 00:13:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     23
    time_act                      0
    Name: 407798, dtype: object
    년                          2018
    월                             9
    일                            18
    요일                            화
    공항                           대구
    출도착                          출발
    상대공항                         김포
    항공사                        에어부산
    항공편                       B1850
    식별번호                   SEw3NzIz
    계획          2019-09-07 20:05:00
    실제          2019-09-07 02:25:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     20
    time_act                      2
    Name: 548327, dtype: object
    년                          2018
    월                             3
    일                             4
    요일                            일
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                         이스타
    항공편                       F1152
    식별번호                   SEw4MDk2
    계획          2019-09-07 23:30:00
    실제          2019-09-07 00:20:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     23
    time_act                      0
    Name: 569782, dtype: object
    년                          2018
    월                             2
    일                             4
    요일                            일
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        아시아나
    항공편                       A1488
    식별번호                   SEw3NzAz
    계획          2019-09-07 22:40:00
    실제          2019-09-07 00:36:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     22
    time_act                      0
    Name: 604997, dtype: object
    년                          2018
    월                             2
    일                             4
    요일                            일
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        아시아나
    항공편                       A1484
    식별번호                   SEw3MjQ3
    계획          2019-09-07 22:00:00
    실제          2019-09-07 00:11:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     22
    time_act                      0
    Name: 604998, dtype: object
    년                          2018
    월                             2
    일                             4
    요일                            일
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        아시아나
    항공편                       A1476
    식별번호                   SEw3Nzcy
    계획          2019-09-07 23:20:00
    실제          2019-09-07 01:14:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     23
    time_act                      1
    Name: 604999, dtype: object
    년                          2018
    월                             1
    일                            12
    요일                            금
    공항                           김해
    출도착                          출발
    상대공항                         청주
    항공사                        제주항공
    항공편                       H1891
    식별번호                   SEw4MjM5
    계획          2019-09-07 23:50:00
    실제          2019-09-07 00:09:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     23
    time_act                      0
    Name: 670119, dtype: object
    년                          2018
    월                             2
    일                             7
    요일                            수
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        제주항공
    항공편                       H1064
    식별번호                   SEw4MDQ5
    계획          2019-09-07 22:50:00
    실제          2019-09-07 00:09:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     22
    time_act                      0
    Name: 673843, dtype: object
    년                          2018
    월                             1
    일                            11
    요일                            목
    공항                           김포
    출도착                          도착
    상대공항                         제주
    항공사                         진에어
    항공편                       I1332
    식별번호                   SEw3NTYx
    계획          2019-09-07 22:25:00
    실제          2019-09-07 02:03:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     22
    time_act                      2
    Name: 723050, dtype: object
    년                          2018
    월                             1
    일                            11
    요일                            목
    공항                           제주
    출도착                          출발
    상대공항                         김포
    항공사                         진에어
    항공편                       I1332
    식별번호                   SEw3NTYx
    계획          2019-09-07 21:10:00
    실제          2019-09-07 01:12:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     21
    time_act                      1
    Name: 723081, dtype: object
    년                          2018
    월                             4
    일                            11
    요일                            수
    공항                           청주
    출도착                          도착
    상대공항                         제주
    항공사                         진에어
    항공편                      I1560A
    식별번호                   SEw3NTY0
    계획          2019-09-07 21:40:00
    실제          2019-09-07 00:14:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     21
    time_act                      0
    Name: 731603, dtype: object
    년                          2018
    월                             1
    일                            12
    요일                            금
    공항                           김포
    출도착                          도착
    상대공항                         제주
    항공사                         티웨이
    항공편                       L1718
    식별번호                   SEw4MjM3
    계획          2019-09-07 22:15:00
    실제          2019-09-07 01:48:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     22
    time_act                      1
    Name: 761425, dtype: object
    년                          2018
    월                             2
    일                             5
    요일                            월
    공항                           대구
    출도착                          도착
    상대공항                         제주
    항공사                         티웨이
    항공편                       L1808
    식별번호                   SEw4MDY5
    계획          2019-09-07 21:55:00
    실제          2019-09-07 00:08:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     21
    time_act                      0
    Name: 763626, dtype: object
    년                          2018
    월                             5
    일                            18
    요일                            금
    공항                           제주
    출도착                          도착
    상대공항                         김포
    항공사                         티웨이
    항공편                       L1735
    식별번호                   SEw4MjM1
    계획          2019-09-07 23:26:00
    실제          2019-09-07 00:28:00
    지연                            Y
    지연사유                        C02
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     23
    time_act                      0
    Name: 772609, dtype: object
    년                          2019
    월                             2
    일                            21
    요일                            목
    공항                           무안
    출도착                          도착
    상대공항                         제주
    항공사                        아시아나
    항공편                       A1168
    식별번호                   SEw3NzM3
    계획          2019-09-07 20:50:00
    실제          2019-09-07 05:45:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     20
    time_act                      5
    Name: 901736, dtype: object
    년                          2019
    월                             5
    일                            18
    요일                            토
    공항                           대구
    출도착                          도착
    상대공항                         제주
    항공사                         티웨이
    항공편                       L1810
    식별번호                   SEw4MjY4
    계획          2019-09-07 22:10:00
    실제          2019-09-07 00:05:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     22
    time_act                      0
    Name: 983578, dtype: object



```python
def check(row):
    if row['실제'].day == today_day+1:
        print(row)
        
train.apply(check, axis = 1)
```

    년                          2017
    월                            10
    일                             2
    요일                            월
    공항                           제주
    출도착                          도착
    상대공항                         청주
    항공사                         진에어
    항공편                       I1559
    식별번호                   SEw3NTYz
    계획          2019-09-07 20:25:00
    실제          2019-09-08 00:05:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     20
    time_act                      0
    Name: 354335, dtype: object
    년                          2018
    월                             2
    일                             5
    요일                            월
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        대한항공
    항공편                       J1707
    식별번호                   SEw3NzY0
    계획          2019-09-07 23:05:00
    실제          2019-09-08 00:13:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     23
    time_act                      0
    Name: 407798, dtype: object
    년                          2018
    월                             9
    일                            18
    요일                            화
    공항                           대구
    출도착                          출발
    상대공항                         김포
    항공사                        에어부산
    항공편                       B1850
    식별번호                   SEw3NzIz
    계획          2019-09-07 20:05:00
    실제          2019-09-08 02:25:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     20
    time_act                      2
    Name: 548327, dtype: object
    년                          2018
    월                             3
    일                             4
    요일                            일
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                         이스타
    항공편                       F1152
    식별번호                   SEw4MDk2
    계획          2019-09-07 23:30:00
    실제          2019-09-08 00:20:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     23
    time_act                      0
    Name: 569782, dtype: object
    년                          2018
    월                             2
    일                             4
    요일                            일
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        아시아나
    항공편                       A1488
    식별번호                   SEw3NzAz
    계획          2019-09-07 22:40:00
    실제          2019-09-08 00:36:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     22
    time_act                      0
    Name: 604997, dtype: object
    년                          2018
    월                             2
    일                             4
    요일                            일
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        아시아나
    항공편                       A1484
    식별번호                   SEw3MjQ3
    계획          2019-09-07 22:00:00
    실제          2019-09-08 00:11:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     22
    time_act                      0
    Name: 604998, dtype: object
    년                          2018
    월                             2
    일                             4
    요일                            일
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        아시아나
    항공편                       A1476
    식별번호                   SEw3Nzcy
    계획          2019-09-07 23:20:00
    실제          2019-09-08 01:14:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     23
    time_act                      1
    Name: 604999, dtype: object
    년                          2018
    월                             1
    일                            12
    요일                            금
    공항                           김해
    출도착                          출발
    상대공항                         청주
    항공사                        제주항공
    항공편                       H1891
    식별번호                   SEw4MjM5
    계획          2019-09-07 23:50:00
    실제          2019-09-08 00:09:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     23
    time_act                      0
    Name: 670119, dtype: object
    년                          2018
    월                             2
    일                             7
    요일                            수
    공항                           제주
    출도착                          출발
    상대공항                         인천
    항공사                        제주항공
    항공편                       H1064
    식별번호                   SEw4MDQ5
    계획          2019-09-07 22:50:00
    실제          2019-09-08 00:09:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          Y
    time_exp                     22
    time_act                      0
    Name: 673843, dtype: object
    년                          2018
    월                             1
    일                            11
    요일                            목
    공항                           김포
    출도착                          도착
    상대공항                         제주
    항공사                         진에어
    항공편                       I1332
    식별번호                   SEw3NTYx
    계획          2019-09-07 22:25:00
    실제          2019-09-08 02:03:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     22
    time_act                      2
    Name: 723050, dtype: object
    년                          2018
    월                             1
    일                            11
    요일                            목
    공항                           제주
    출도착                          출발
    상대공항                         김포
    항공사                         진에어
    항공편                       I1332
    식별번호                   SEw3NTYx
    계획          2019-09-07 21:10:00
    실제          2019-09-08 01:12:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     21
    time_act                      1
    Name: 723081, dtype: object
    년                          2018
    월                             4
    일                            11
    요일                            수
    공항                           청주
    출도착                          도착
    상대공항                         제주
    항공사                         진에어
    항공편                      I1560A
    식별번호                   SEw3NTY0
    계획          2019-09-07 21:40:00
    실제          2019-09-08 00:14:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     21
    time_act                      0
    Name: 731603, dtype: object
    년                          2018
    월                             1
    일                            12
    요일                            금
    공항                           김포
    출도착                          도착
    상대공항                         제주
    항공사                         티웨이
    항공편                       L1718
    식별번호                   SEw4MjM3
    계획          2019-09-07 22:15:00
    실제          2019-09-08 01:48:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     22
    time_act                      1
    Name: 761425, dtype: object
    년                          2018
    월                             2
    일                             5
    요일                            월
    공항                           대구
    출도착                          도착
    상대공항                         제주
    항공사                         티웨이
    항공편                       L1808
    식별번호                   SEw4MDY5
    계획          2019-09-07 21:55:00
    실제          2019-09-08 00:08:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     21
    time_act                      0
    Name: 763626, dtype: object
    년                          2018
    월                             5
    일                            18
    요일                            금
    공항                           제주
    출도착                          도착
    상대공항                         김포
    항공사                         티웨이
    항공편                       L1735
    식별번호                   SEw4MjM1
    계획          2019-09-07 23:26:00
    실제          2019-09-08 00:28:00
    지연                            Y
    지연사유                        C02
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     23
    time_act                      0
    Name: 772609, dtype: object
    년                          2019
    월                             2
    일                            21
    요일                            목
    공항                           무안
    출도착                          도착
    상대공항                         제주
    항공사                        아시아나
    항공편                       A1168
    식별번호                   SEw3NzM3
    계획          2019-09-07 20:50:00
    실제          2019-09-08 05:45:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     20
    time_act                      5
    Name: 901736, dtype: object
    년                          2019
    월                             5
    일                            18
    요일                            토
    공항                           대구
    출도착                          도착
    상대공항                         제주
    항공사                         티웨이
    항공편                       L1810
    식별번호                   SEw4MjY4
    계획          2019-09-07 22:10:00
    실제          2019-09-08 00:05:00
    지연                            N
    지연사유                        NaN
    결항                            N
    결항사유                        NaN
    부정기편                          N
    time_exp                     22
    time_act                      0
    Name: 983578, dtype: object





    0         None
    1         None
    2         None
    3         None
    4         None
    5         None
    6         None
    7         None
    8         None
    9         None
    10        None
    11        None
    12        None
    13        None
    14        None
    15        None
    16        None
    17        None
    18        None
    19        None
    20        None
    21        None
    22        None
    23        None
    24        None
    25        None
    26        None
    27        None
    28        None
    29        None
              ... 
    987679    None
    987680    None
    987681    None
    987682    None
    987683    None
    987684    None
    987685    None
    987686    None
    987687    None
    987688    None
    987689    None
    987690    None
    987691    None
    987692    None
    987693    None
    987694    None
    987695    None
    987696    None
    987697    None
    987698    None
    987699    None
    987700    None
    987701    None
    987702    None
    987703    None
    987704    None
    987705    None
    987706    None
    987707    None
    987708    None
    Length: 979450, dtype: object




```python
train["차이"] = train["실제"] - train["계획"]
train["차이_minus"] = train["계획"] - train["실제"]


train['차이'][train['차이'].astype('timedelta64[m]') > 0 ] =   train['차이'].astype('timedelta64[m]') 
train['차이'][train['차이_minus'].astype('timedelta64[m]') > 0 ] = 0 - train['차이_minus'].astype('timedelta64[m]') 

train.head(100)

```

    /Users/Choiwansik/anaconda3/envs/kagglecom/lib/python3.6/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    /Users/Choiwansik/anaconda3/envs/kagglecom/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>년</th>
      <th>월</th>
      <th>일</th>
      <th>요일</th>
      <th>공항</th>
      <th>출도착</th>
      <th>상대공항</th>
      <th>항공사</th>
      <th>항공편</th>
      <th>식별번호</th>
      <th>...</th>
      <th>실제</th>
      <th>지연</th>
      <th>지연사유</th>
      <th>결항</th>
      <th>결항사유</th>
      <th>부정기편</th>
      <th>time_exp</th>
      <th>time_act</th>
      <th>차이</th>
      <th>차이_minus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>...</td>
      <td>2019-09-07 10:32:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>10</td>
      <td>10</td>
      <td>27.0</td>
      <td>-1 days +23:33:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1954</td>
      <td>SEw3NzE4</td>
      <td>...</td>
      <td>2019-09-07 09:31:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>9</td>
      <td>9</td>
      <td>1.0</td>
      <td>-1 days +23:59:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1956</td>
      <td>SEw3NzE4</td>
      <td>...</td>
      <td>2019-09-07 13:03:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>12</td>
      <td>13</td>
      <td>18.0</td>
      <td>-1 days +23:42:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>...</td>
      <td>2019-09-07 14:09:00</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>13</td>
      <td>14</td>
      <td>44.0</td>
      <td>-1 days +23:16:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1958</td>
      <td>SEw3NzE4</td>
      <td>...</td>
      <td>2019-09-07 16:31:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>16</td>
      <td>16</td>
      <td>21.0</td>
      <td>-1 days +23:39:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1959</td>
      <td>SEw3NTk5</td>
      <td>...</td>
      <td>2019-09-07 17:21:00</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>16</td>
      <td>17</td>
      <td>36.0</td>
      <td>-1 days +23:24:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1960</td>
      <td>SEw3NTk5</td>
      <td>...</td>
      <td>2019-09-07 19:43:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>19</td>
      <td>19</td>
      <td>13.0</td>
      <td>-1 days +23:47:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1961</td>
      <td>SEw3NTk5</td>
      <td>...</td>
      <td>2019-09-07 20:52:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>20</td>
      <td>20</td>
      <td>17.0</td>
      <td>-1 days +23:43:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김해</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1015</td>
      <td>SEw3NzA2</td>
      <td>...</td>
      <td>2019-09-07 17:03:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>17</td>
      <td>17</td>
      <td>-2.0</td>
      <td>00:02:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1242</td>
      <td>SEw3NzA2</td>
      <td>...</td>
      <td>2019-09-07 20:36:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>20</td>
      <td>20</td>
      <td>11.0</td>
      <td>-1 days +23:49:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1257</td>
      <td>SEw3NzA4</td>
      <td>...</td>
      <td>2019-09-07 12:44:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>12</td>
      <td>12</td>
      <td>4.0</td>
      <td>-1 days +23:56:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1220</td>
      <td>SEw3NzA4</td>
      <td>...</td>
      <td>2019-09-07 13:41:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>13</td>
      <td>13</td>
      <td>16.0</td>
      <td>-1 days +23:44:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1203</td>
      <td>SEw3NzA4</td>
      <td>...</td>
      <td>2019-09-07 08:03:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>8</td>
      <td>8</td>
      <td>-2.0</td>
      <td>00:02:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>대구</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1813</td>
      <td>SEw3NzA4</td>
      <td>...</td>
      <td>2019-09-07 17:36:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>17</td>
      <td>17</td>
      <td>-9.0</td>
      <td>00:09:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>대구</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1814</td>
      <td>SEw3NzA4</td>
      <td>...</td>
      <td>2019-09-07 18:35:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>18</td>
      <td>18</td>
      <td>10.0</td>
      <td>-1 days +23:50:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1254</td>
      <td>SEw3NzA4</td>
      <td>...</td>
      <td>2019-09-07 09:24:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>9</td>
      <td>9</td>
      <td>19.0</td>
      <td>-1 days +23:41:00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김해</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1021</td>
      <td>SEw3NzA4</td>
      <td>...</td>
      <td>2019-09-07 20:59:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>21</td>
      <td>20</td>
      <td>-6.0</td>
      <td>00:06:00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1235</td>
      <td>SEw3NzE2</td>
      <td>...</td>
      <td>2019-09-07 19:29:00</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>18</td>
      <td>19</td>
      <td>34.0</td>
      <td>-1 days +23:26:00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1258</td>
      <td>SEw3NTY2</td>
      <td>...</td>
      <td>2019-09-07 11:43:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>11</td>
      <td>11</td>
      <td>13.0</td>
      <td>-1 days +23:47:00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1236</td>
      <td>SEw3NzE2</td>
      <td>...</td>
      <td>2019-09-07 13:25:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>13</td>
      <td>13</td>
      <td>10.0</td>
      <td>-1 days +23:50:00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1248</td>
      <td>SEw3NzE2</td>
      <td>...</td>
      <td>2019-09-07 21:16:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>21</td>
      <td>21</td>
      <td>16.0</td>
      <td>-1 days +23:44:00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>광주</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1909</td>
      <td>SEw3NzE2</td>
      <td>...</td>
      <td>2019-09-07 15:55:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>15</td>
      <td>15</td>
      <td>10.0</td>
      <td>-1 days +23:50:00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>광주</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1908</td>
      <td>SEw3NzE2</td>
      <td>...</td>
      <td>2019-09-07 16:53:00</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>16</td>
      <td>16</td>
      <td>33.0</td>
      <td>-1 days +23:27:00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>대구</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1806</td>
      <td>SEw4MjQx</td>
      <td>...</td>
      <td>2019-09-07 09:50:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>9</td>
      <td>9</td>
      <td>20.0</td>
      <td>-1 days +23:40:00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>원주</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1855</td>
      <td>SEw4MjQx</td>
      <td>...</td>
      <td>2019-09-07 12:14:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>12</td>
      <td>12</td>
      <td>-1.0</td>
      <td>00:01:00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>원주</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1854</td>
      <td>SEw4MjQx</td>
      <td>...</td>
      <td>2019-09-07 13:20:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>13</td>
      <td>13</td>
      <td>10.0</td>
      <td>-1 days +23:50:00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>사천</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1835</td>
      <td>SEw4MjQx</td>
      <td>...</td>
      <td>2019-09-07 16:13:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>16</td>
      <td>16</td>
      <td>-2.0</td>
      <td>00:02:00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1219</td>
      <td>SEw3NTY2</td>
      <td>...</td>
      <td>2019-09-07 14:33:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>14</td>
      <td>14</td>
      <td>8.0</td>
      <td>-1 days +23:52:00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1253</td>
      <td>SEw3NzE2</td>
      <td>...</td>
      <td>2019-09-07 08:14:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>8</td>
      <td>8</td>
      <td>-1.0</td>
      <td>00:01:00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>사천</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1834</td>
      <td>SEw4MjQx</td>
      <td>...</td>
      <td>2019-09-07 17:18:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>17</td>
      <td>17</td>
      <td>8.0</td>
      <td>-1 days +23:52:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1240</td>
      <td>SEw3NTI1</td>
      <td>...</td>
      <td>2019-09-07 18:19:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>18</td>
      <td>18</td>
      <td>14.0</td>
      <td>-1 days +23:46:00</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1243</td>
      <td>SEw3NTI1</td>
      <td>...</td>
      <td>2019-09-07 21:19:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>21</td>
      <td>21</td>
      <td>-1.0</td>
      <td>00:01:00</td>
    </tr>
    <tr>
      <th>72</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1222</td>
      <td>SEw3NTI1</td>
      <td>...</td>
      <td>2019-09-07 13:46:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>13</td>
      <td>13</td>
      <td>11.0</td>
      <td>-1 days +23:49:00</td>
    </tr>
    <tr>
      <th>73</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1224</td>
      <td>SEw3NzUy</td>
      <td>...</td>
      <td>2019-09-07 14:26:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>14</td>
      <td>14</td>
      <td>21.0</td>
      <td>-1 days +23:39:00</td>
    </tr>
    <tr>
      <th>74</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1231</td>
      <td>SEw3NzUy</td>
      <td>...</td>
      <td>2019-09-07 18:07:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>17</td>
      <td>18</td>
      <td>12.0</td>
      <td>-1 days +23:48:00</td>
    </tr>
    <tr>
      <th>75</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1212</td>
      <td>SEw3NTI1</td>
      <td>...</td>
      <td>2019-09-07 09:14:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>9</td>
      <td>9</td>
      <td>14.0</td>
      <td>-1 days +23:46:00</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1227</td>
      <td>SEw3NTI1</td>
      <td>...</td>
      <td>2019-09-07 16:44:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>16</td>
      <td>16</td>
      <td>4.0</td>
      <td>-1 days +23:56:00</td>
    </tr>
    <tr>
      <th>77</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김해</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1017</td>
      <td>SEw3NTQw</td>
      <td>...</td>
      <td>2019-09-07 19:04:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>19</td>
      <td>19</td>
      <td>-1.0</td>
      <td>00:01:00</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김해</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1014</td>
      <td>SEw3NTk5</td>
      <td>...</td>
      <td>2019-09-07 15:15:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>15</td>
      <td>15</td>
      <td>10.0</td>
      <td>-1 days +23:50:00</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>청주</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1959</td>
      <td>SEw3NTk5</td>
      <td>...</td>
      <td>2019-09-07 18:07:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>17</td>
      <td>18</td>
      <td>17.0</td>
      <td>-1 days +23:43:00</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>청주</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1960</td>
      <td>SEw3NTk5</td>
      <td>...</td>
      <td>2019-09-07 18:54:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>18</td>
      <td>18</td>
      <td>24.0</td>
      <td>-1 days +23:36:00</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>청주</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1961</td>
      <td>SEw3NTk5</td>
      <td>...</td>
      <td>2019-09-07 21:39:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>21</td>
      <td>21</td>
      <td>-1.0</td>
      <td>00:01:00</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김해</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1011</td>
      <td>SEw3NzI0</td>
      <td>...</td>
      <td>2019-09-07 15:02:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>14</td>
      <td>15</td>
      <td>7.0</td>
      <td>-1 days +23:53:00</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1218</td>
      <td>SEw3NzI0</td>
      <td>...</td>
      <td>2019-09-07 12:33:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>12</td>
      <td>12</td>
      <td>23.0</td>
      <td>-1 days +23:37:00</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1208</td>
      <td>SEw3NzI0</td>
      <td>...</td>
      <td>2019-09-07 08:57:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>8</td>
      <td>8</td>
      <td>17.0</td>
      <td>-1 days +23:43:00</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1207</td>
      <td>SEw3NzI0</td>
      <td>...</td>
      <td>2019-09-07 11:42:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>11</td>
      <td>11</td>
      <td>12.0</td>
      <td>-1 days +23:48:00</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1221</td>
      <td>SEw3NTQw</td>
      <td>...</td>
      <td>2019-09-07 14:55:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>14</td>
      <td>14</td>
      <td>10.0</td>
      <td>-1 days +23:50:00</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1230</td>
      <td>SEw3NTQw</td>
      <td>...</td>
      <td>2019-09-07 15:57:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>15</td>
      <td>15</td>
      <td>12.0</td>
      <td>-1 days +23:48:00</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김해</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1010</td>
      <td>SEw3NTQw</td>
      <td>...</td>
      <td>2019-09-07 10:54:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>10</td>
      <td>10</td>
      <td>14.0</td>
      <td>-1 days +23:46:00</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김해</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1020</td>
      <td>SEw3NzI1</td>
      <td>...</td>
      <td>2019-09-07 19:18:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>19</td>
      <td>19</td>
      <td>13.0</td>
      <td>-1 days +23:47:00</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김해</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1023</td>
      <td>SEw3NzI1</td>
      <td>...</td>
      <td>2019-09-07 21:37:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>21</td>
      <td>21</td>
      <td>2.0</td>
      <td>-1 days +23:58:00</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1215</td>
      <td>SEw3NzI3</td>
      <td>...</td>
      <td>2019-09-07 12:53:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>12</td>
      <td>12</td>
      <td>3.0</td>
      <td>-1 days +23:57:00</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1249</td>
      <td>SEw3NzI3</td>
      <td>...</td>
      <td>2019-09-07 22:27:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>22</td>
      <td>22</td>
      <td>17.0</td>
      <td>-1 days +23:43:00</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1205</td>
      <td>SEw3NzI3</td>
      <td>...</td>
      <td>2019-09-07 09:07:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>9</td>
      <td>9</td>
      <td>2.0</td>
      <td>-1 days +23:58:00</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>광주</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1911</td>
      <td>SEw3NzI3</td>
      <td>...</td>
      <td>2019-09-07 18:48:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>18</td>
      <td>18</td>
      <td>-2.0</td>
      <td>00:02:00</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1206</td>
      <td>SEw3NzU3</td>
      <td>...</td>
      <td>2019-09-07 08:16:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>8</td>
      <td>8</td>
      <td>16.0</td>
      <td>-1 days +23:44:00</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1211</td>
      <td>SEw3NzU3</td>
      <td>...</td>
      <td>2019-09-07 11:26:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>11</td>
      <td>11</td>
      <td>6.0</td>
      <td>-1 days +23:54:00</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1209</td>
      <td>SEw3NzU4</td>
      <td>...</td>
      <td>2019-09-07 11:48:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>11</td>
      <td>11</td>
      <td>13.0</td>
      <td>-1 days +23:47:00</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>대구</td>
      <td>도착</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1811</td>
      <td>SEw4MjI0</td>
      <td>...</td>
      <td>2019-09-07 15:47:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>16</td>
      <td>15</td>
      <td>-13.0</td>
      <td>00:13:00</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>광주</td>
      <td>출발</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1910</td>
      <td>SEw3NzI3</td>
      <td>...</td>
      <td>2019-09-07 19:47:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>19</td>
      <td>19</td>
      <td>12.0</td>
      <td>-1 days +23:48:00</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 21 columns</p>
</div>




```python
train.columns
```




    Index(['년', '월', '일', '요일', '공항', '출도착', '상대공항', '항공사', '항공편', '식별번호', '계획',
           '실제', '지연', '지연사유', '결항', '결항사유', '부정기편', 'time_exp', 'time_act', '차이',
           '차이_minus'],
          dtype='object')




```python
idx_1 = train[((train['차이'] > 30) | (train['차이'] < -30.0)) & (train['지연'] == 'N')].index
train = train.drop(idx_1,axis = 0)
idx_2 = train[((train['차이'] <= 30) & (train['차이'] >= -30.0)) & (train['지연'] == 'Y')].index
train = train.drop(idx_2,axis = 0)
train.shape
```




    (978734, 21)




```python
remove_variable = ['time_exp','time_act','차이','차이_minus']
train.drop(remove_variable, axis = 1, inplace = True)
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>년</th>
      <th>월</th>
      <th>일</th>
      <th>요일</th>
      <th>공항</th>
      <th>출도착</th>
      <th>상대공항</th>
      <th>항공사</th>
      <th>항공편</th>
      <th>식별번호</th>
      <th>계획</th>
      <th>실제</th>
      <th>지연</th>
      <th>지연사유</th>
      <th>결항</th>
      <th>결항사유</th>
      <th>부정기편</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 10:05:00</td>
      <td>2019-09-07 10:32:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1954</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 09:30:00</td>
      <td>2019-09-07 09:31:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1956</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 12:45:00</td>
      <td>2019-09-07 13:03:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 13:25:00</td>
      <td>2019-09-07 14:09:00</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1958</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 16:10:00</td>
      <td>2019-09-07 16:31:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
train['계획'] = train['계획'].apply(lambda x : str(x)[11:16])
train['실제'] = train['실제'].apply(lambda x : str(x)[11:16])
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>년</th>
      <th>월</th>
      <th>일</th>
      <th>요일</th>
      <th>공항</th>
      <th>출도착</th>
      <th>상대공항</th>
      <th>항공사</th>
      <th>항공편</th>
      <th>식별번호</th>
      <th>계획</th>
      <th>실제</th>
      <th>지연</th>
      <th>지연사유</th>
      <th>결항</th>
      <th>결항사유</th>
      <th>부정기편</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>10:05</td>
      <td>10:32</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1954</td>
      <td>SEw3NzE4</td>
      <td>09:30</td>
      <td>09:31</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1956</td>
      <td>SEw3NzE4</td>
      <td>12:45</td>
      <td>13:03</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>13:25</td>
      <td>14:09</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1958</td>
      <td>SEw3NzE4</td>
      <td>16:10</td>
      <td>16:31</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.shape
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>년</th>
      <th>월</th>
      <th>일</th>
      <th>요일</th>
      <th>공항</th>
      <th>출도착</th>
      <th>상대공항</th>
      <th>항공사</th>
      <th>항공편</th>
      <th>식별번호</th>
      <th>계획</th>
      <th>실제</th>
      <th>지연</th>
      <th>지연사유</th>
      <th>결항</th>
      <th>결항사유</th>
      <th>부정기편</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>10:05</td>
      <td>10:32</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1954</td>
      <td>SEw3NzE4</td>
      <td>09:30</td>
      <td>09:31</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1956</td>
      <td>SEw3NzE4</td>
      <td>12:45</td>
      <td>13:03</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>출발</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>13:25</td>
      <td>14:09</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>도착</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1958</td>
      <td>SEw3NzE4</td>
      <td>16:10</td>
      <td>16:31</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.to_csv('train_preprocessed.csv', encoding = "CP949")
```


```python

```
