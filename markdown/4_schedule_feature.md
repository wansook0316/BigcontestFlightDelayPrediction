
# 특정 일에 일련 번호 별 운행한 Group indexing (index)

# 그 index를 기준으로 출발시간 별 순서 추가 (Order)


```python
import pandas as pd 
import numpy as np
# 시각화 준비
import matplotlib
import matplotlib.pyplot as plt
# Jupyter Notebook 내부에 그래프를 출력하도록 설정
%matplotlib inline
import datetime

train = pd.read_csv("../input/AFSNT.csv", engine = "python", encoding = 'CP949')


train['STT'] = train['STT'].apply(lambda x: 60 * np.float64(x.split(':')[0]) + np.float64(x.split(':')[1]))
train['ATT'] = train['ATT'].apply(lambda x: 60 * np.float64(x.split(':')[0]) + np.float64(x.split(':')[1]))
print(train.head())

train['DELAY'] = train['ATT'] - train['STT']


variables_to_remove = ['DRR','CNL',"CNR"]
train.drop(variables_to_remove, axis = 1, inplace = True)

train.dropna(inplace=True)
missing_train = train.isnull().sum(axis=0).reset_index()
missing_train.columns = ['variable', 'missing values']
missing_train['filling factor (%)']=(train.shape[0]-missing_train['missing values'])/train.shape[0]*100
missing_train.sort_values('filling factor (%)').reset_index(drop = True)

train_col_name = list(train.columns)
train_col_name.extend(['ORDER', 'index'])

train_order = pd.DataFrame(columns=train_col_name)
print(train)

print(type(train['SDT_YY'][0]))

start_day = datetime.date(2017,1,1)
last_day = datetime.date(2019,6,30)
reg_count = 0
ok = 0

while start_day <= last_day:
    train_ = train[(train['SDT_YY'] == start_day.year) & (train['SDT_MM'] == start_day.month) & (train['SDT_DD'] == start_day.day)]
    reg_name = train_['REG'].unique()
    for i in range(len(reg_name)):
        temp = train_[train_['REG'] == reg_name[i]].sort_values(['STT'])
        temp = np.array(temp)
        temp = np.hstack((temp, np.arange(0,temp.shape[0],1).reshape(temp.shape[0],1)))
        temp = np.hstack((temp, reg_count * np.ones((temp.shape[0])).reshape(temp.shape[0],1)))
        reg_count += 1
                         
        if ok == 0:
            train_order = temp
            ok = 1
        else:
            train_order = np.vstack((train_order, temp))
            
    
    start_day += datetime.timedelta(1)
    print(start_day)


#train_order[train_order['index'] == 0]

train_order = pd.DataFrame(train_order,columns=train_col_name)
print(train_order)

train.head()

train = train_order
train


```

       SDT_YY  SDT_MM  SDT_DD SDT_DY   ARP   ODP FLO    FLT       REG AOD IRR  \
    0    2017       1       1      일  ARP3  ARP6   J  J1955  SEw3NzE4   D   N   
    1    2017       1       1      일  ARP3  ARP6   J  J1954  SEw3NzE4   A   N   
    2    2017       1       1      일  ARP3  ARP6   J  J1956  SEw3NzE4   A   N   
    3    2017       1       1      일  ARP3  ARP6   J  J1957  SEw3NzE4   D   N   
    4    2017       1       1      일  ARP3  ARP6   J  J1958  SEw3NzE4   A   N   
    
         STT    ATT DLY  DRR CNL  CNR  
    0  605.0  632.0   N  NaN   N  NaN  
    1  570.0  571.0   N  NaN   N  NaN  
    2  765.0  783.0   N  NaN   N  NaN  
    3  805.0  849.0   Y  C02   N  NaN  
    4  970.0  991.0   N  NaN   N  NaN  
            SDT_YY  SDT_MM  SDT_DD SDT_DY    ARP   ODP FLO    FLT       REG AOD  \
    0         2017       1       1      일   ARP3  ARP6   J  J1955  SEw3NzE4   D   
    1         2017       1       1      일   ARP3  ARP6   J  J1954  SEw3NzE4   A   
    2         2017       1       1      일   ARP3  ARP6   J  J1956  SEw3NzE4   A   
    3         2017       1       1      일   ARP3  ARP6   J  J1957  SEw3NzE4   D   
    4         2017       1       1      일   ARP3  ARP6   J  J1958  SEw3NzE4   A   
    5         2017       1       1      일   ARP3  ARP6   J  J1959  SEw3NTk5   D   
    6         2017       1       1      일   ARP3  ARP6   J  J1960  SEw3NTk5   A   
    7         2017       1       1      일   ARP3  ARP6   J  J1961  SEw3NTk5   D   
    8         2017       1       1      일   ARP2  ARP3   J  J1015  SEw3NzA2   A   
    9         2017       1       1      일   ARP1  ARP3   J  J1242  SEw3NzA2   D   
    10        2017       1       1      일   ARP1  ARP3   J  J1257  SEw3NzA4   A   
    11        2017       1       1      일   ARP1  ARP3   J  J1220  SEw3NzA4   D   
    12        2017       1       1      일   ARP1  ARP3   J  J1203  SEw3NzA4   A   
    13        2017       1       1      일   ARP4  ARP3   J  J1813  SEw3NzA4   A   
    14        2017       1       1      일   ARP4  ARP3   J  J1814  SEw3NzA4   D   
    15        2017       1       1      일   ARP1  ARP3   J  J1254  SEw3NzA4   D   
    16        2017       1       1      일   ARP2  ARP3   J  J1021  SEw3NzA4   A   
    17        2017       1       1      일   ARP1  ARP3   J  J1235  SEw3NzE2   A   
    18        2017       1       1      일   ARP1  ARP3   J  J1258  SEw3NTY2   D   
    19        2017       1       1      일   ARP1  ARP3   J  J1236  SEw3NzE2   D   
    20        2017       1       1      일   ARP1  ARP3   J  J1248  SEw3NzE2   D   
    21        2017       1       1      일   ARP8  ARP3   J  J1909  SEw3NzE2   A   
    22        2017       1       1      일   ARP8  ARP3   J  J1908  SEw3NzE2   D   
    23        2017       1       1      일   ARP4  ARP3   J  J1806  SEw4MjQx   D   
    24        2017       1       1      일  ARP14  ARP3   J  J1855  SEw4MjQx   A   
    25        2017       1       1      일  ARP14  ARP3   J  J1854  SEw4MjQx   D   
    26        2017       1       1      일  ARP12  ARP3   J  J1835  SEw4MjQx   A   
    27        2017       1       1      일   ARP1  ARP3   J  J1219  SEw3NTY2   A   
    28        2017       1       1      일   ARP1  ARP3   J  J1253  SEw3NzE2   A   
    29        2017       1       1      일  ARP12  ARP3   J  J1834  SEw4MjQx   D   
    ...        ...     ...     ...    ...    ...   ...  ..    ...       ...  ..   
    987679    2019       6      30      일   ARP3  ARP1   L  L1736  SEw4MzIz   D   
    987680    2019       6      30      일   ARP3  ARP1   L  L1720  SEw4MzIz   D   
    987681    2019       6      30      일   ARP3  ARP1   L  L1713  SEw4MzIz   A   
    987682    2019       6      30      일   ARP3  ARP1   L  L1705  SEw4MzIz   A   
    987683    2019       6      30      일   ARP3  ARP1   L  L1708  SEw4MzIz   D   
    987684    2019       6      30      일   ARP3  ARP1   L  L1728  SEw4MzIz   D   
    987685    2019       6      30      일   ARP3  ARP1   L  L1735  SEw4MzIz   A   
    987686    2019       6      30      일   ARP3  ARP1   L  L1725  SEw4MzIz   A   
    987687    2019       6      30      일   ARP3  ARP1   L  L1701  SEw4MzU0   A   
    987688    2019       6      30      일   ARP3  ARP1   L  L1703  SEw4MzU0   A   
    987689    2019       6      30      일   ARP3  ARP1   L  L1702  SEw4MzU0   D   
    987690    2019       6      30      일   ARP3  ARP1   L  L1704  SEw4MjM3   D   
    987691    2019       6      30      일   ARP3  ARP8   L  L1901  SEw4MDk4   A   
    987692    2019       6      30      일   ARP3  ARP8   L  L1902  SEw4MDk4   D   
    987693    2019       6      30      일   ARP3  ARP8   L  L1904  SEw4MDAw   D   
    987694    2019       6      30      일   ARP3  ARP8   L  L1903  SEw4MDAw   A   
    987695    2019       6      30      일   ARP3  ARP8   L  L1906  SEw4MDAw   D   
    987696    2019       6      30      일   ARP3  ARP8   L  L1905  SEw4MDAw   A   
    987697    2019       6      30      일   ARP3  ARP7   L  L1931  SEw4MDk4   A   
    987698    2019       6      30      일   ARP3  ARP7   L  L1932  SEw4MzU0   D   
    987699    2019       6      30      일   ARP3  ARP4   L  L1805  SEw4MDU2   A   
    987700    2019       6      30      일   ARP3  ARP4   L  L1802  SEw4MDU2   D   
    987701    2019       6      30      일   ARP3  ARP4   L  L1806  SEw4MDAw   D   
    987702    2019       6      30      일   ARP3  ARP4   L  L1803  SEw4MDAw   A   
    987703    2019       6      30      일   ARP3  ARP4   L  L1808  SEw4MzYz   D   
    987704    2019       6      30      일   ARP3  ARP4   L  L1810  SEw4MjM1   D   
    987705    2019       6      30      일   ARP3  ARP4   L  L1809  SEw4MjM1   A   
    987706    2019       6      30      일   ARP3  ARP4   L  L1801  SEw4MzYz   A   
    987707    2019       6      30      일   ARP3  ARP4   L  L1807  SEw4MjM3   A   
    987708    2019       6      30      일   ARP3  ARP4   L  L1804  SEw4MjM3   D   
    
           IRR     STT     ATT DLY  DELAY  
    0        N   605.0   632.0   N   27.0  
    1        N   570.0   571.0   N    1.0  
    2        N   765.0   783.0   N   18.0  
    3        N   805.0   849.0   Y   44.0  
    4        N   970.0   991.0   N   21.0  
    5        N  1005.0  1041.0   Y   36.0  
    6        N  1170.0  1183.0   N   13.0  
    7        N  1235.0  1252.0   N   17.0  
    8        N  1025.0  1023.0   N   -2.0  
    9        N  1225.0  1236.0   N   11.0  
    10       N   760.0   764.0   N    4.0  
    11       N   805.0   821.0   N   16.0  
    12       N   485.0   483.0   N   -2.0  
    13       N  1065.0  1056.0   N   -9.0  
    14       N  1105.0  1115.0   N   10.0  
    15       N   545.0   564.0   N   19.0  
    16       N  1265.0  1259.0   N   -6.0  
    17       N  1135.0  1169.0   Y   34.0  
    18       N   690.0   703.0   N   13.0  
    19       N   795.0   805.0   N   10.0  
    20       N  1260.0  1276.0   N   16.0  
    21       N   945.0   955.0   N   10.0  
    22       N   980.0  1013.0   Y   33.0  
    23       N   570.0   590.0   N   20.0  
    24       N   735.0   734.0   N   -1.0  
    25       N   790.0   800.0   N   10.0  
    26       N   975.0   973.0   N   -2.0  
    27       N   865.0   873.0   N    8.0  
    28       N   495.0   494.0   N   -1.0  
    29       N  1030.0  1038.0   N    8.0  
    ...     ..     ...     ...  ..    ...  
    987679   N  1265.0  1279.0   N   14.0  
    987680   N   810.0   823.0   N   13.0  
    987681   N   775.0   768.0   N   -7.0  
    987682   N   565.0   553.0   N  -12.0  
    987683   N   600.0   615.0   N   15.0  
    987684   N  1025.0  1056.0   Y   31.0  
    987685   N  1200.0  1227.0   N   27.0  
    987686   N   985.0   998.0   N   13.0  
    987687   Y   720.0   711.0   N   -9.0  
    987688   Y  1030.0  1043.0   N   13.0  
    987689   Y   755.0   767.0   N   12.0  
    987690   Y  1080.0  1116.0   Y   36.0  
    987691   N   625.0   608.0   N  -17.0  
    987692   N   485.0   495.0   N   10.0  
    987693   N   650.0   668.0   N   18.0  
    987694   N   775.0   777.0   N    2.0  
    987695   N   805.0   829.0   N   24.0  
    987696   N   930.0   939.0   N    9.0  
    987697   N   450.0   436.0   N  -14.0  
    987698   N  1090.0  1113.0   N   23.0  
    987699   N   665.0   649.0   N  -16.0  
    987700   N   505.0   515.0   N   10.0  
    987701   N  1085.0  1115.0   N   30.0  
    987702   N   595.0   589.0   N   -6.0  
    987703   N  1150.0  1164.0   N   14.0  
    987704   N  1270.0  1313.0   Y   43.0  
    987705   N  1235.0  1259.0   N   24.0  
    987706   N   440.0   430.0   N  -10.0  
    987707   N  1050.0  1052.0   N    2.0  
    987708   N   885.0   894.0   N    9.0  
    
    [979446 rows x 15 columns]
    <class 'numpy.int64'>
    2017-01-02
    2017-01-03
    2017-01-04
    2017-01-05
    2017-01-06
    2017-01-07
    2017-01-08
    2017-01-09
    2017-01-10
    2017-01-11
    2017-01-12
    2017-01-13
    2017-01-14
    2017-01-15
    2017-01-16
    2017-01-17
    2017-01-18
    2017-01-19
    2017-01-20
    2017-01-21
    2017-01-22
    2017-01-23
    2017-01-24
    2017-01-25
    2017-01-26
    2017-01-27
    2017-01-28
    2017-01-29
    2017-01-30
    2017-01-31
    2017-02-01
    2017-02-02
    2017-02-03
    2017-02-04
    2017-02-05
    2017-02-06
    2017-02-07
    2017-02-08
    2017-02-09
    2017-02-10
    2017-02-11
    2017-02-12
    2017-02-13
    2017-02-14
    2017-02-15
    2017-02-16
    2017-02-17
    2017-02-18
    2017-02-19
    2017-02-20
    2017-02-21
    2017-02-22
    2017-02-23
    2017-02-24
    2017-02-25
    2017-02-26
    2017-02-27
    2017-02-28
    2017-03-01
    2017-03-02
    2017-03-03
    2017-03-04
    2017-03-05
    2017-03-06
    2017-03-07
    2017-03-08
    2017-03-09
    2017-03-10
    2017-03-11
    2017-03-12
    2017-03-13
    2017-03-14
    2017-03-15
    2017-03-16
    2017-03-17
    2017-03-18
    2017-03-19
    2017-03-20
    2017-03-21
    2017-03-22
    2017-03-23
    2017-03-24
    2017-03-25
    2017-03-26
    2017-03-27
    2017-03-28
    2017-03-29
    2017-03-30
    2017-03-31
    2017-04-01
    2017-04-02
    2017-04-03
    2017-04-04
    2017-04-05
    2017-04-06
    2017-04-07
    2017-04-08
    2017-04-09
    2017-04-10
    2017-04-11
    2017-04-12
    2017-04-13
    2017-04-14
    2017-04-15
    2017-04-16
    2017-04-17
    2017-04-18
    2017-04-19
    2017-04-20
    2017-04-21
    2017-04-22
    2017-04-23
    2017-04-24
    2017-04-25
    2017-04-26
    2017-04-27
    2017-04-28
    2017-04-29
    2017-04-30
    2017-05-01
    2017-05-02
    2017-05-03
    2017-05-04
    2017-05-05
    2017-05-06
    2017-05-07
    2017-05-08
    2017-05-09
    2017-05-10
    2017-05-11
    2017-05-12
    2017-05-13
    2017-05-14
    2017-05-15
    2017-05-16
    2017-05-17
    2017-05-18
    2017-05-19
    2017-05-20
    2017-05-21
    2017-05-22
    2017-05-23
    2017-05-24
    2017-05-25
    2017-05-26
    2017-05-27
    2017-05-28
    2017-05-29
    2017-05-30
    2017-05-31
    2017-06-01
    2017-06-02
    2017-06-03
    2017-06-04
    2017-06-05
    2017-06-06
    2017-06-07
    2017-06-08
    2017-06-09
    2017-06-10
    2017-06-11
    2017-06-12
    2017-06-13
    2017-06-14
    2017-06-15
    2017-06-16
    2017-06-17
    2017-06-18
    2017-06-19
    2017-06-20
    2017-06-21
    2017-06-22
    2017-06-23
    2017-06-24
    2017-06-25
    2017-06-26
    2017-06-27
    2017-06-28
    2017-06-29
    2017-06-30
    2017-07-01
    2017-07-02
    2017-07-03
    2017-07-04
    2017-07-05
    2017-07-06
    2017-07-07
    2017-07-08
    2017-07-09
    2017-07-10
    2017-07-11
    2017-07-12
    2017-07-13
    2017-07-14
    2017-07-15
    2017-07-16
    2017-07-17
    2017-07-18
    2017-07-19
    2017-07-20
    2017-07-21
    2017-07-22
    2017-07-23
    2017-07-24
    2017-07-25
    2017-07-26
    2017-07-27
    2017-07-28
    2017-07-29
    2017-07-30
    2017-07-31
    2017-08-01
    2017-08-02
    2017-08-03
    2017-08-04
    2017-08-05
    2017-08-06
    2017-08-07
    2017-08-08
    2017-08-09
    2017-08-10
    2017-08-11
    2017-08-12
    2017-08-13
    2017-08-14
    2017-08-15
    2017-08-16
    2017-08-17
    2017-08-18
    2017-08-19
    2017-08-20
    2017-08-21
    2017-08-22
    2017-08-23
    2017-08-24
    2017-08-25
    2017-08-26
    2017-08-27
    2017-08-28
    2017-08-29
    2017-08-30
    2017-08-31
    2017-09-01
    2017-09-02
    2017-09-03
    2017-09-04
    2017-09-05
    2017-09-06
    2017-09-07
    2017-09-08
    2017-09-09
    2017-09-10
    2017-09-11
    2017-09-12
    2017-09-13
    2017-09-14
    2017-09-15
    2017-09-16
    2017-09-17
    2017-09-18
    2017-09-19
    2017-09-20
    2017-09-21
    2017-09-22
    2017-09-23
    2017-09-24
    2017-09-25
    2017-09-26
    2017-09-27
    2017-09-28
    2017-09-29
    2017-09-30
    2017-10-01
    2017-10-02
    2017-10-03
    2017-10-04
    2017-10-05
    2017-10-06
    2017-10-07
    2017-10-08
    2017-10-09
    2017-10-10
    2017-10-11
    2017-10-12
    2017-10-13
    2017-10-14
    2017-10-15
    2017-10-16
    2017-10-17
    2017-10-18
    2017-10-19
    2017-10-20
    2017-10-21
    2017-10-22
    2017-10-23
    2017-10-24
    2017-10-25
    2017-10-26
    2017-10-27
    2017-10-28
    2017-10-29
    2017-10-30
    2017-10-31
    2017-11-01
    2017-11-02
    2017-11-03
    2017-11-04
    2017-11-05
    2017-11-06
    2017-11-07
    2017-11-08
    2017-11-09
    2017-11-10
    2017-11-11
    2017-11-12
    2017-11-13
    2017-11-14
    2017-11-15
    2017-11-16
    2017-11-17
    2017-11-18
    2017-11-19
    2017-11-20
    2017-11-21
    2017-11-22
    2017-11-23
    2017-11-24
    2017-11-25
    2017-11-26
    2017-11-27
    2017-11-28
    2017-11-29
    2017-11-30
    2017-12-01
    2017-12-02
    2017-12-03
    2017-12-04
    2017-12-05
    2017-12-06
    2017-12-07
    2017-12-08
    2017-12-09
    2017-12-10
    2017-12-11
    2017-12-12
    2017-12-13
    2017-12-14
    2017-12-15
    2017-12-16
    2017-12-17
    2017-12-18
    2017-12-19
    2017-12-20
    2017-12-21
    2017-12-22
    2017-12-23
    2017-12-24
    2017-12-25
    2017-12-26
    2017-12-27
    2017-12-28
    2017-12-29
    2017-12-30
    2017-12-31
    2018-01-01
    2018-01-02
    2018-01-03
    2018-01-04
    2018-01-05
    2018-01-06
    2018-01-07
    2018-01-08
    2018-01-09
    2018-01-10
    2018-01-11
    2018-01-12
    2018-01-13
    2018-01-14
    2018-01-15
    2018-01-16
    2018-01-17
    2018-01-18
    2018-01-19
    2018-01-20
    2018-01-21
    2018-01-22
    2018-01-23
    2018-01-24
    2018-01-25
    2018-01-26
    2018-01-27
    2018-01-28
    2018-01-29
    2018-01-30
    2018-01-31
    2018-02-01
    2018-02-02
    2018-02-03
    2018-02-04
    2018-02-05
    2018-02-06
    2018-02-07
    2018-02-08
    2018-02-09
    2018-02-10
    2018-02-11
    2018-02-12
    2018-02-13
    2018-02-14
    2018-02-15
    2018-02-16
    2018-02-17
    2018-02-18
    2018-02-19
    2018-02-20
    2018-02-21
    2018-02-22
    2018-02-23
    2018-02-24
    2018-02-25
    2018-02-26
    2018-02-27
    2018-02-28
    2018-03-01
    2018-03-02
    2018-03-03
    2018-03-04
    2018-03-05
    2018-03-06
    2018-03-07
    2018-03-08
    2018-03-09
    2018-03-10
    2018-03-11
    2018-03-12
    2018-03-13
    2018-03-14
    2018-03-15
    2018-03-16
    2018-03-17
    2018-03-18
    2018-03-19
    2018-03-20
    2018-03-21
    2018-03-22
    2018-03-23
    2018-03-24
    2018-03-25
    2018-03-26
    2018-03-27
    2018-03-28
    2018-03-29
    2018-03-30
    2018-03-31
    2018-04-01
    2018-04-02
    2018-04-03
    2018-04-04
    2018-04-05
    2018-04-06
    2018-04-07
    2018-04-08
    2018-04-09
    2018-04-10
    2018-04-11
    2018-04-12
    2018-04-13
    2018-04-14
    2018-04-15
    2018-04-16
    2018-04-17
    2018-04-18
    2018-04-19
    2018-04-20
    2018-04-21
    2018-04-22
    2018-04-23
    2018-04-24
    2018-04-25
    2018-04-26
    2018-04-27
    2018-04-28
    2018-04-29
    2018-04-30
    2018-05-01
    2018-05-02
    2018-05-03
    2018-05-04
    2018-05-05
    2018-05-06
    2018-05-07
    2018-05-08
    2018-05-09
    2018-05-10
    2018-05-11
    2018-05-12
    2018-05-13
    2018-05-14
    2018-05-15
    2018-05-16
    2018-05-17
    2018-05-18
    2018-05-19
    2018-05-20
    2018-05-21
    2018-05-22
    2018-05-23
    2018-05-24
    2018-05-25
    2018-05-26
    2018-05-27
    2018-05-28
    2018-05-29
    2018-05-30
    2018-05-31
    2018-06-01
    2018-06-02
    2018-06-03
    2018-06-04
    2018-06-05
    2018-06-06
    2018-06-07
    2018-06-08
    2018-06-09
    2018-06-10
    2018-06-11
    2018-06-12
    2018-06-13
    2018-06-14
    2018-06-15
    2018-06-16
    2018-06-17
    2018-06-18
    2018-06-19
    2018-06-20
    2018-06-21
    2018-06-22
    2018-06-23
    2018-06-24
    2018-06-25
    2018-06-26
    2018-06-27
    2018-06-28
    2018-06-29
    2018-06-30
    2018-07-01
    2018-07-02
    2018-07-03
    2018-07-04
    2018-07-05
    2018-07-06
    2018-07-07
    2018-07-08
    2018-07-09
    2018-07-10
    2018-07-11
    2018-07-12
    2018-07-13
    2018-07-14
    2018-07-15
    2018-07-16
    2018-07-17
    2018-07-18
    2018-07-19
    2018-07-20
    2018-07-21
    2018-07-22
    2018-07-23
    2018-07-24
    2018-07-25
    2018-07-26
    2018-07-27
    2018-07-28
    2018-07-29
    2018-07-30
    2018-07-31
    2018-08-01
    2018-08-02
    2018-08-03
    2018-08-04
    2018-08-05
    2018-08-06
    2018-08-07
    2018-08-08
    2018-08-09
    2018-08-10
    2018-08-11
    2018-08-12
    2018-08-13
    2018-08-14
    2018-08-15
    2018-08-16
    2018-08-17
    2018-08-18
    2018-08-19
    2018-08-20
    2018-08-21
    2018-08-22
    2018-08-23
    2018-08-24
    2018-08-25
    2018-08-26
    2018-08-27
    2018-08-28
    2018-08-29
    2018-08-30
    2018-08-31
    2018-09-01
    2018-09-02
    2018-09-03
    2018-09-04
    2018-09-05
    2018-09-06
    2018-09-07
    2018-09-08
    2018-09-09
    2018-09-10
    2018-09-11
    2018-09-12


# index 를 기준으로 계획시간과 항공편이 같은 것을 schedule 이라 가정
# schedule 추가


```python
station = []
station_stt = []

index = train['index'].unique()

for idx in index:
    temp1 = list(train[(train['index'] == idx)]['FLT'])
    temp2 = list(train[(train['index'] == idx)]['STT'])

    if temp1 in station:
        continue
    else:
        station.append(temp1)
        station_stt.append(temp2)
        
schedule = []

for idx in index:
    temp = list(train[(train['index'] == idx)]['FLT'])
    position = station.index(temp)
    for j in range(len(temp)):
        schedule.append(position)
train['schedule'] = schedule
```

# schedule 별 min, max, count, mean 추출

# order 별 평균


```python
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

schedule_ = train['schedule'].unique()

train['schedule_min'] = 0
train['schedule_max'] = 0
train['schedule_count'] = 0
train['schedule_mean'] = 0

for i in schedule_:
    temp = train[train['schedule'] == i]
    stats = temp['DELAY'].groupby(temp['STT']).apply(get_stats).unstack()
    schedule_stt = temp['STT'].unique()
    for j in schedule_stt:
        idx = temp[temp['STT'] == j].index
        train['schedule_min'].loc[idx] = stats.loc[j][0]
        train['schedule_max'].loc[idx] = stats.loc[j][1]
        train['schedule_count'].loc[idx] = stats.loc[j][2]
        train['schedule_mean'].loc[idx] = stats.loc[j][3]
```


```python
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
      <th>...</th>
      <th>CNL</th>
      <th>CNR</th>
      <th>DELAY</th>
      <th>ORDER</th>
      <th>index</th>
      <th>schedule</th>
      <th>schedule_min</th>
      <th>schedule_max</th>
      <th>schedule_count</th>
      <th>schedule_mean</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 25 columns</p>
</div>



# test data에 입히기


```python
test = pd.read_csv("../input/AFSNT_DLY.csv", engine = "python")
test = test.sort_values('STT')
test['STT'] = test['STT'].apply(lambda x : np.float64(x.split(':')[0])*60 + np.float64(x.split(':')[1]))
test.head()
test['ORDER'] = -1
test['order_mean'] = -1
test['schedule_min'] = -1
test['schedule_max'] = -1
test['schedule_count'] = -1
test['schedule_mean'] = -1
test['schedule'] = -1
test.head()
```


```python
start_day = datetime.date(2019, 9, 16)
end_day = datetime.date(2019, 9, 30)

while start_day <= end_day:
    print(start_day)
    temp = test[(test['SDT_YY'] == start_day.year) & (test['SDT_MM'] == start_day.month) & (test['SDT_DD'] == start_day.day)]

    temp = temp[['FLT', 'STT']]
    temp = temp.sort_values('STT')
    
    idx1 = temp.index
    idx2 = range(len(idx1))
    idx = zip(idx2, idx1)
    idx = dict(idx)
    
    temp = list(temp['FLT'])
    
    for i in range(len(station)):
        start = 0
        temp_idx1 = []
        temp_idx2 = []
        for j in range(0, len(station[i])):
            if station[i][j] in temp[start:]:
                temp_idx1.append(temp[start:].index(station[i][j]) + start)
                temp_idx2.append(station[i][j])
                start = temp[start:].index(station[i][j]) + start + 1
        if station[i] == temp_idx2:
            #print(station[i])
            #print(temp_idx2)
            add_idx = []
            for j in range(len(station[i])):
                add_idx.append(idx[temp_idx1[j]])
            if list(test['STT'].loc[add_idx]) == station_stt[i]:
                #print(add_idx)
                test['schedule'].loc[add_idx] = i
                
                ran = len(train[train['schedule'] == i]['ORDER'].unique())
                temp_data = train[train['schedule'] == i].copy()
                temp_data = temp_data.reset_index()
                temp_data = temp_data.loc[0:ran - 1]
                
                test['ORDER'].loc[add_idx] = np.array(temp_data['ORDER'])
                test['order_mean'].loc[add_idx] = np.array(temp_data['order_mean'])
                test['schedule_min'].loc[add_idx] = np.array(temp_data['schedule_min'])
                test['schedule_max'].loc[add_idx] = np.array(temp_data['schedule_max'])
                test['schedule_count'].loc[add_idx] = np.array(temp_data['schedule_count'])
                test['schedule_mean'].loc[add_idx] = np.array(temp_data['schedule_mean'])
                #print(test['schedule'].loc[add_idx])
    start_day = start_day + datetime.timedelta(1)
    
test
```




```python

```


```python

```


```python

```


```python

```


```python

```

# schedule 파일 쓰기


```python
train_order.to_csv("../train_add_schedule.csv", encoding='cp949')
test.to_csv("../pred_add_schedule.csv", endcoding = 'cp949')
```


```python

```
