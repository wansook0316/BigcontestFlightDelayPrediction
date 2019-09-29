
# 라이브러리 불러오기


```python
import datetime, warnings, scipy 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
import os
import datetime

from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"
pd.options.display.max_columns = 50
%matplotlib inline
warnings.filterwarnings("ignore")

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
    
mpl.rcParams['axes.unicode_minus'] = False   
#그래프에서 마이너스 기호가 표시되도록 하는 설정
```


```python
# 훈련 데이터를 읽어온다

train = pd.read_csv("../input/AFSNT.CSV", engine = "python", encoding ='CP949')
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
      <th>SDT_YY</th>
      <th>SDT_MM</th>
      <th>SDT_DD</th>
      <th>SDT_DY</th>
      <th>ARP</th>
      <th>ODP</th>
      <th>FLO</th>
      <th>FLT</th>
      <th>REG</th>
      <th>AOD</th>
      <th>IRR</th>
      <th>STT</th>
      <th>ATT</th>
      <th>DLY</th>
      <th>DRR</th>
      <th>CNL</th>
      <th>CNR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>D</td>
      <td>N</td>
      <td>10:05</td>
      <td>10:32</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1954</td>
      <td>SEw3NzE4</td>
      <td>A</td>
      <td>N</td>
      <td>9:30</td>
      <td>9:31</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1956</td>
      <td>SEw3NzE4</td>
      <td>A</td>
      <td>N</td>
      <td>12:45</td>
      <td>13:03</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>D</td>
      <td>N</td>
      <td>13:25</td>
      <td>14:09</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1958</td>
      <td>SEw3NzE4</td>
      <td>A</td>
      <td>N</td>
      <td>16:10</td>
      <td>16:31</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 987709 entries, 0 to 987708
    Data columns (total 17 columns):
    SDT_YY    987709 non-null int64
    SDT_MM    987709 non-null int64
    SDT_DD    987709 non-null int64
    SDT_DY    987709 non-null object
    ARP       987709 non-null object
    ODP       987709 non-null object
    FLO       987709 non-null object
    FLT       987709 non-null object
    REG       979446 non-null object
    AOD       987709 non-null object
    IRR       987709 non-null object
    STT       987709 non-null object
    ATT       987709 non-null object
    DLY       987709 non-null object
    DRR       118937 non-null object
    CNL       987709 non-null object
    CNR       8259 non-null object
    dtypes: int64(3), object(14)
    memory usage: 128.1+ MB


# 같은 항공편이 출도착으로 나뉘어 row로 존재하므로 이를 합친다.


```python
# 출발하는 데이터/ 도착하는 데이터로 구분
trainA = train[train.AOD == "A"]
print(len(trainA))
trainD = train[train.AOD == "D"]
print(len(trainD))
```

    493992
    493717



```python
# 년, 월, 일, 요일, 항공사, 항공편, 식별번호로 묶어서 데이터 행 합치기
all_df = pd.merge(left=trainD, right=trainA, how='inner',on=['SDT_YY','SDT_MM','SDT_DD', "SDT_DY", "FLO", "FLT", "REG"], sort=False)
all_df.head()
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
      <th>SDT_YY</th>
      <th>SDT_MM</th>
      <th>SDT_DD</th>
      <th>SDT_DY</th>
      <th>ARP_x</th>
      <th>ODP_x</th>
      <th>FLO</th>
      <th>FLT</th>
      <th>REG</th>
      <th>AOD_x</th>
      <th>IRR_x</th>
      <th>STT_x</th>
      <th>ATT_x</th>
      <th>DLY_x</th>
      <th>DRR_x</th>
      <th>CNL_x</th>
      <th>CNR_x</th>
      <th>ARP_y</th>
      <th>ODP_y</th>
      <th>AOD_y</th>
      <th>IRR_y</th>
      <th>STT_y</th>
      <th>ATT_y</th>
      <th>DLY_y</th>
      <th>DRR_y</th>
      <th>CNL_y</th>
      <th>CNR_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>D</td>
      <td>N</td>
      <td>10:05</td>
      <td>10:32</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP6</td>
      <td>ARP3</td>
      <td>A</td>
      <td>N</td>
      <td>11:10</td>
      <td>11:18</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>D</td>
      <td>N</td>
      <td>13:25</td>
      <td>14:09</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP6</td>
      <td>ARP3</td>
      <td>A</td>
      <td>N</td>
      <td>14:30</td>
      <td>14:56</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1959</td>
      <td>SEw3NTk5</td>
      <td>D</td>
      <td>N</td>
      <td>16:45</td>
      <td>17:21</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP6</td>
      <td>ARP3</td>
      <td>A</td>
      <td>N</td>
      <td>17:50</td>
      <td>18:07</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1961</td>
      <td>SEw3NTk5</td>
      <td>D</td>
      <td>N</td>
      <td>20:35</td>
      <td>20:52</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP6</td>
      <td>ARP3</td>
      <td>A</td>
      <td>N</td>
      <td>21:40</td>
      <td>21:39</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP1</td>
      <td>ARP3</td>
      <td>J</td>
      <td>J1242</td>
      <td>SEw3NzA2</td>
      <td>D</td>
      <td>N</td>
      <td>20:25</td>
      <td>20:36</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP3</td>
      <td>ARP1</td>
      <td>A</td>
      <td>N</td>
      <td>21:30</td>
      <td>21:27</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_df.shape
```




    (492436, 27)




```python
all_df.head()
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
      <th>SDT_YY</th>
      <th>SDT_MM</th>
      <th>SDT_DD</th>
      <th>SDT_DY</th>
      <th>ARP_x</th>
      <th>ODP_x</th>
      <th>FLO</th>
      <th>FLT</th>
      <th>REG</th>
      <th>AOD_x</th>
      <th>IRR_x</th>
      <th>STT_x</th>
      <th>ATT_x</th>
      <th>DLY_x</th>
      <th>DRR_x</th>
      <th>CNL_x</th>
      <th>CNR_x</th>
      <th>ARP_y</th>
      <th>ODP_y</th>
      <th>AOD_y</th>
      <th>IRR_y</th>
      <th>STT_y</th>
      <th>ATT_y</th>
      <th>DLY_y</th>
      <th>DRR_y</th>
      <th>CNL_y</th>
      <th>CNR_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>D</td>
      <td>N</td>
      <td>10:05</td>
      <td>10:32</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP6</td>
      <td>ARP3</td>
      <td>A</td>
      <td>N</td>
      <td>11:10</td>
      <td>11:18</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>D</td>
      <td>N</td>
      <td>13:25</td>
      <td>14:09</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP6</td>
      <td>ARP3</td>
      <td>A</td>
      <td>N</td>
      <td>14:30</td>
      <td>14:56</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1959</td>
      <td>SEw3NTk5</td>
      <td>D</td>
      <td>N</td>
      <td>16:45</td>
      <td>17:21</td>
      <td>Y</td>
      <td>C02</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP6</td>
      <td>ARP3</td>
      <td>A</td>
      <td>N</td>
      <td>17:50</td>
      <td>18:07</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1961</td>
      <td>SEw3NTk5</td>
      <td>D</td>
      <td>N</td>
      <td>20:35</td>
      <td>20:52</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP6</td>
      <td>ARP3</td>
      <td>A</td>
      <td>N</td>
      <td>21:40</td>
      <td>21:39</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP1</td>
      <td>ARP3</td>
      <td>J</td>
      <td>J1242</td>
      <td>SEw3NzA2</td>
      <td>D</td>
      <td>N</td>
      <td>20:25</td>
      <td>20:36</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
      <td>ARP3</td>
      <td>ARP1</td>
      <td>A</td>
      <td>N</td>
      <td>21:30</td>
      <td>21:27</td>
      <td>N</td>
      <td>NaN</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# 결항 데이터를 없앤다.


```python
## 결항된 데이터 제거를 위해 결항 갯수 확인

print(len(all_df))
all_df_x = all_df[all_df.CNL_x == "Y"]
print(len(all_df_x))
all_df_y = all_df[all_df.CNL_y == "Y"]
print(len(all_df_y))
```

    492436
    4079
    4079



```python
## 결항 데이터 제거

all_df = all_df[all_df.CNL_x == "N"]
print(len(all_df))

# 확인
print(all_df[all_df.CNL_x =="N"].count())
print(all_df[all_df.CNL_y =="N"].count())

```

    488357
    SDT_YY    488357
    SDT_MM    488357
    SDT_DD    488357
    SDT_DY    488357
    ARP_x     488357
    ODP_x     488357
    FLO       488357
    FLT       488357
    REG       488357
    AOD_x     488357
    IRR_x     488357
    STT_x     488357
    ATT_x     488357
    DLY_x     488357
    DRR_x      85645
    CNL_x     488357
    CNR_x          0
    ARP_y     488357
    ODP_y     488357
    AOD_y     488357
    IRR_y     488357
    STT_y     488357
    ATT_y     488357
    DLY_y     488357
    DRR_y      32720
    CNL_y     488357
    CNR_y          0
    dtype: int64
    SDT_YY    488357
    SDT_MM    488357
    SDT_DD    488357
    SDT_DY    488357
    ARP_x     488357
    ODP_x     488357
    FLO       488357
    FLT       488357
    REG       488357
    AOD_x     488357
    IRR_x     488357
    STT_x     488357
    ATT_x     488357
    DLY_x     488357
    DRR_x      85645
    CNL_x     488357
    CNR_x          0
    ARP_y     488357
    ODP_y     488357
    AOD_y     488357
    IRR_y     488357
    STT_y     488357
    ATT_y     488357
    DLY_y     488357
    DRR_y      32720
    CNL_y     488357
    CNR_y          0
    dtype: int64



```python
all_df.shape
```




    (488357, 27)



# 변수 제외 및 이름을 바꾼다.


```python
# 필요없는 변수 제외하기
# 총 10개
# 겹치는 것 'ARP_y','ODP_y' = 공항, 상대공항 - 하나만 있으면 충분
# 필요없는 것 'AOD_x','AOD_y' = 출도착 여부 /'IRR_x','IRR_y' = 부정기편 여부 / 'CNL_x','CNR_x','CNL_y','CNR_y' = 결항관련 항목들
remove_variable = ['AOD_x','IRR_x','ARP_y','ODP_y','AOD_y','IRR_y','CNL_x','CNR_x','CNL_y','CNR_y']
all_df.drop(remove_variable, axis = 1, inplace = True)
all_df.head()
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
      <th>SDT_YY</th>
      <th>SDT_MM</th>
      <th>SDT_DD</th>
      <th>SDT_DY</th>
      <th>ARP_x</th>
      <th>ODP_x</th>
      <th>FLO</th>
      <th>FLT</th>
      <th>REG</th>
      <th>STT_x</th>
      <th>ATT_x</th>
      <th>DLY_x</th>
      <th>DRR_x</th>
      <th>STT_y</th>
      <th>ATT_y</th>
      <th>DLY_y</th>
      <th>DRR_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>10:05</td>
      <td>10:32</td>
      <td>N</td>
      <td>NaN</td>
      <td>11:10</td>
      <td>11:18</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>13:25</td>
      <td>14:09</td>
      <td>Y</td>
      <td>C02</td>
      <td>14:30</td>
      <td>14:56</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1959</td>
      <td>SEw3NTk5</td>
      <td>16:45</td>
      <td>17:21</td>
      <td>Y</td>
      <td>C02</td>
      <td>17:50</td>
      <td>18:07</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP3</td>
      <td>ARP6</td>
      <td>J</td>
      <td>J1961</td>
      <td>SEw3NTk5</td>
      <td>20:35</td>
      <td>20:52</td>
      <td>N</td>
      <td>NaN</td>
      <td>21:40</td>
      <td>21:39</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>ARP1</td>
      <td>ARP3</td>
      <td>J</td>
      <td>J1242</td>
      <td>SEw3NzA2</td>
      <td>20:25</td>
      <td>20:36</td>
      <td>N</td>
      <td>NaN</td>
      <td>21:30</td>
      <td>21:27</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
## 변수명 및 값들 한국어로 바꾸기

print('Dataframe dimensions:', all_df.shape)
all_df.columns = ['년','월','일', '요일', '공항', '상대공항', '항공사', '항공편', '식별번호', '예정출발시간',
             '실제출발시간','출발편지연여부', '출발편지연사유', '예정도착시간', '실제도착시간', '도착편지연여부', '도착편지연사유', ]
#recollist = ['년','월','일', '요일', '공항', '출도착', '상대공항', '항공사', '항공편', '식별번호', 
#              '계획', '실제', '지연', '지연사유', '결항', '결항사유', '부정기편']
#all_df = all_df[recollist]
transairport = {'ARP1':'김포', 'ARP2':'김해', 'ARP3': '제주', 'ARP4':'대구', 'ARP5': '울산',
           'ARP6':'청주', 'ARP7':'무안', 'ARP8':'광주', 'ARP9':'여수', 'ARP10':'양양',
           'ARP11':'포항', 'ARP12':'사천', 'ARP13':'군산', 'ARP14':'원주', 'ARP15':'인천'}
all_df['공항'] = all_df['공항'].apply(lambda x: transairport[x])
all_df['상대공항'] = all_df['상대공항'].apply(lambda x: transairport[x])
#train_inner["출도착"] = train_inner["출도착"].apply(lambda x: "출발" if x=="D" else x)
#train_inner["출도착"] = train_inner["출도착"].apply(lambda x: "도착" if x=="A" else x)
#df = df.iloc[df['항공사']!= ['C', 'D', 'E', 'G', 'K'],]
transairline = {'A':'아시아나', 'B':'에어부산', 'C': '전일본항공ANA', 'D':'에어서울', 'E': '불명',
           'F':'이스타', 'G':'일본항공', 'H':'제주항공', 'I':'진에어', 'J':'대한항공',
           'K':'타이완', 'L':'티웨이', 'M':'신규'}
all_df['항공사'] = all_df['항공사'].apply(lambda x: transairline[x])
```

    Dataframe dimensions: (488357, 17)



```python
all_df.head()
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
      <th>상대공항</th>
      <th>항공사</th>
      <th>항공편</th>
      <th>식별번호</th>
      <th>예정출발시간</th>
      <th>실제출발시간</th>
      <th>출발편지연여부</th>
      <th>출발편지연사유</th>
      <th>예정도착시간</th>
      <th>실제도착시간</th>
      <th>도착편지연여부</th>
      <th>도착편지연사유</th>
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
      <td>청주</td>
      <td>대한항공</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>10:05</td>
      <td>10:32</td>
      <td>N</td>
      <td>NaN</td>
      <td>11:10</td>
      <td>11:18</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>13:25</td>
      <td>14:09</td>
      <td>Y</td>
      <td>C02</td>
      <td>14:30</td>
      <td>14:56</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1959</td>
      <td>SEw3NTk5</td>
      <td>16:45</td>
      <td>17:21</td>
      <td>Y</td>
      <td>C02</td>
      <td>17:50</td>
      <td>18:07</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1961</td>
      <td>SEw3NTk5</td>
      <td>20:35</td>
      <td>20:52</td>
      <td>N</td>
      <td>NaN</td>
      <td>21:40</td>
      <td>21:39</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1242</td>
      <td>SEw3NzA2</td>
      <td>20:25</td>
      <td>20:36</td>
      <td>N</td>
      <td>NaN</td>
      <td>21:30</td>
      <td>21:27</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# 다음날로 넘어가는 항공편들 x_차이를 구하기 위한 작업들


```python
all_df['예정출발시간'] = pd.to_datetime(all_df['예정출발시간'])
all_df['실제출발시간'] = pd.to_datetime(all_df['실제출발시간'])
all_df['예정도착시간'] = pd.to_datetime(all_df['예정도착시간'])
all_df['실제도착시간'] = pd.to_datetime(all_df['실제도착시간'])
```


```python
all_df['time_exp_dep'] = all_df['예정출발시간'].apply(lambda x : x.time().hour)
all_df['time_act_dep'] = all_df['실제출발시간'].apply(lambda x : x.time().hour)
all_df['time_exp_ari'] = all_df['예정도착시간'].apply(lambda x : x.time().hour)
all_df['time_act_ari'] = all_df['실제도착시간'].apply(lambda x : x.time().hour)

def nextday_time_mapping_dep(row):
  if (row['time_exp_dep'] >= 20) and ((row['time_act_dep'] >=0) and (row['time_act_dep'] <= 5)):
    return row['실제출발시간'] + timedelta(days = +1)
  else:
    return row['실제출발시간']

def nextday_time_mapping_ari(row):
  if (row['time_exp_ari'] >= 20.0) and ((row['time_act_ari'] >=0) and (row['time_act_ari'] <= 5)):
    return row['실제도착시간'] + timedelta(days = +1)
  else: 
    return row['실제도착시간']
```


```python
print(pd.to_datetime('today').day)

today_day = pd.to_datetime('today').day

all_df['실제출발시간'] = all_df.apply(nextday_time_mapping_dep, axis = 1)
all_df['실제도착시간'] = all_df.apply(nextday_time_mapping_ari, axis = 1)

```

    7



```python
def check_dep(row):
#     print(row['실제출발시간'].day)
#     print(row)
    if row['실제출발시간'].day == today_day+1:
        print(row['실제출발시간'].day)

def check_ari(row):
#     print(row['실제출발시간'].day)
#     print(row)
    if row['실제도착시간'].day == today_day+1:
        print(row['실제도착시간'].day)

all_df.apply(check_dep, axis = 1)
all_df.apply(check_ari, axis = 1)

```

    8
    8
    8
    8
    8
    8
    8
    8
    8
    8
    8
    8





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
    492406    None
    492407    None
    492408    None
    492409    None
    492410    None
    492411    None
    492412    None
    492413    None
    492414    None
    492415    None
    492416    None
    492417    None
    492418    None
    492419    None
    492420    None
    492421    None
    492422    None
    492423    None
    492424    None
    492425    None
    492426    None
    492427    None
    492428    None
    492429    None
    492430    None
    492431    None
    492432    None
    492433    None
    492434    None
    492435    None
    Length: 488357, dtype: object




```python
## 지역시간x , 지연시간 y , 계획 비행 시간 차이, 실제 비행 시간 차이 구하기 
all_df["x_차이"] = all_df["실제출발시간"] - all_df["예정출발시간"]
all_df["y_차이"] = all_df["실제도착시간"] - all_df["예정도착시간"]
all_df["plan_차이"] = all_df["예정도착시간"] - all_df["예정출발시간"]
all_df["true_차이"] = all_df["실제도착시간"] - all_df["실제출발시간"]
## - 값을 따로 넣어주기 위해 양수, 음수 값 변환
all_df["x_차이_minus"] = all_df["예정출발시간"] - all_df["실제출발시간"]
all_df["y_차이_minus"] = all_df["예정도착시간"] - all_df["실제도착시간"]
all_df["plan_차이_minus"] = all_df["예정출발시간"] - all_df["예정도착시간"]
all_df["true_차이_minus"] = all_df["실제출발시간"] - all_df["실제도착시간"]
## 더 빨리 도착한 값은 -를 붙여서 변환, 지연된 값은 그대로 저장
all_df['x_차이'][all_df['x_차이'].astype('timedelta64[m]') > 0 ] =   all_df['x_차이'].astype('timedelta64[m]') 
all_df['x_차이'][all_df['x_차이_minus'].astype('timedelta64[m]') > 0 ] = 0 - all_df['x_차이_minus'].astype('timedelta64[m]') 
all_df['y_차이'][all_df['y_차이'].astype('timedelta64[m]') > 0 ] =   all_df['y_차이'].astype('timedelta64[m]') 
all_df['y_차이'][all_df['y_차이_minus'].astype('timedelta64[m]') > 0 ] = 0 - all_df['y_차이_minus'].astype('timedelta64[m]') 
all_df['plan_차이'][all_df['plan_차이'].astype('timedelta64[m]') > 0 ] =   all_df['plan_차이'].astype('timedelta64[m]') 
all_df['plan_차이'][all_df['plan_차이_minus'].astype('timedelta64[m]') > 0 ] = 0 - all_df['plan_차이_minus'].astype('timedelta64[m]') 
all_df['true_차이'][all_df['true_차이'].astype('timedelta64[m]') > 0 ] =   all_df['true_차이'].astype('timedelta64[m]') 
all_df['true_차이'][all_df['true_차이_minus'].astype('timedelta64[m]') > 0 ] = 0 - all_df['true_차이_minus'].astype('timedelta64[m]') 
## minus 칼럼 제거 
all_df = all_df.drop(["x_차이_minus", "y_차이_minus", "plan_차이_minus","true_차이_minus"], axis = 1)

```

# x_차이 튀는 값들 제거


```python

```


```python
idx_1 = all_df[((all_df['x_차이'] > 30) | (all_df['x_차이'] < -30.0)) & (all_df['출발편지연여부'] == 'N')].index
all_df = all_df.drop(idx_1,axis = 0)
idx_2 = all_df[((all_df['x_차이'] <= 30) & (all_df['x_차이'] >= -30.0)) & (all_df['출발편지연여부'] == 'Y')].index
all_df = all_df.drop(idx_2,axis = 0)
idx_3 = all_df[((all_df['y_차이'] > 30) | (all_df['y_차이'] < -30.0)) & (all_df['도착편지연여부'] == 'N')].index
all_df = all_df.drop(idx_3,axis = 0)
idx_4 = all_df[((all_df['y_차이'] <= 30) & (all_df['y_차이'] >= -30.0)) & (all_df['도착편지연여부'] == 'Y')].index
all_df = all_df.drop(idx_4,axis = 0)
all_df.shape
```




    (487695, 25)




```python

```


```python

```


```python

```

# 비행시간 차이 feature 만들기


```python
all_df["비행시간차이"] = all_df["plan_차이"] - all_df["true_차이"]

all_df.head()
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
      <th>상대공항</th>
      <th>항공사</th>
      <th>항공편</th>
      <th>식별번호</th>
      <th>예정출발시간</th>
      <th>실제출발시간</th>
      <th>출발편지연여부</th>
      <th>출발편지연사유</th>
      <th>예정도착시간</th>
      <th>실제도착시간</th>
      <th>도착편지연여부</th>
      <th>도착편지연사유</th>
      <th>time_exp_dep</th>
      <th>time_act_dep</th>
      <th>time_exp_ari</th>
      <th>time_act_ari</th>
      <th>x_차이</th>
      <th>y_차이</th>
      <th>plan_차이</th>
      <th>true_차이</th>
      <th>비행시간차이</th>
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
      <td>청주</td>
      <td>대한항공</td>
      <td>J1955</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 10:05:00</td>
      <td>2019-09-07 10:32:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>2019-09-07 11:10:00</td>
      <td>2019-09-07 11:18:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>10</td>
      <td>10</td>
      <td>11</td>
      <td>11</td>
      <td>27.0</td>
      <td>8.0</td>
      <td>65.0</td>
      <td>46.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1957</td>
      <td>SEw3NzE4</td>
      <td>2019-09-07 13:25:00</td>
      <td>2019-09-07 14:09:00</td>
      <td>Y</td>
      <td>C02</td>
      <td>2019-09-07 14:30:00</td>
      <td>2019-09-07 14:56:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>13</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>44.0</td>
      <td>26.0</td>
      <td>65.0</td>
      <td>47.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1959</td>
      <td>SEw3NTk5</td>
      <td>2019-09-07 16:45:00</td>
      <td>2019-09-07 17:21:00</td>
      <td>Y</td>
      <td>C02</td>
      <td>2019-09-07 17:50:00</td>
      <td>2019-09-07 18:07:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>16</td>
      <td>17</td>
      <td>17</td>
      <td>18</td>
      <td>36.0</td>
      <td>17.0</td>
      <td>65.0</td>
      <td>46.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>제주</td>
      <td>청주</td>
      <td>대한항공</td>
      <td>J1961</td>
      <td>SEw3NTk5</td>
      <td>2019-09-07 20:35:00</td>
      <td>2019-09-07 20:52:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>2019-09-07 21:40:00</td>
      <td>2019-09-07 21:39:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>20</td>
      <td>20</td>
      <td>21</td>
      <td>21</td>
      <td>17.0</td>
      <td>-1.0</td>
      <td>65.0</td>
      <td>47.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>1</td>
      <td>1</td>
      <td>일</td>
      <td>김포</td>
      <td>제주</td>
      <td>대한항공</td>
      <td>J1242</td>
      <td>SEw3NzA2</td>
      <td>2019-09-07 20:25:00</td>
      <td>2019-09-07 20:36:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>2019-09-07 21:30:00</td>
      <td>2019-09-07 21:27:00</td>
      <td>N</td>
      <td>NaN</td>
      <td>20</td>
      <td>20</td>
      <td>21</td>
      <td>21</td>
      <td>11.0</td>
      <td>-3.0</td>
      <td>65.0</td>
      <td>51.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_df.to_csv('all_df.csv', encoding = "CP949")
```
