# titanic
    まだ試行錯誤中のクソザコ
    モデル量産大会
## use column

- Pclass (0-1正規化)
- Sex (female=0, male=1)
- Age (100で割る)
- sibsp (10で割る)
- parch (10で割る)
- fare (logを取ってから0-1正規化 最大512.3292 最少0.0 logを取る時0は0のままにしておく)
- Embarked(C=0.0 Q=0.5 S=1.0)


## Data argumentation

```
`>>> import pandas as pd
>>> df = pd.to_csv("train.csv")

>>> df.isnull().any(axis=0)
PassengerId.    False
Survived       False
Pclass         False
Name           False
Sex            False
Age             True
SibSp          Falsec
Parch          False
Ticket         False
Fare           False
Cabin           True
Embarked        True
dtype: bool
>>> df.isnull().sum()
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64

>>> df = df.drop("Embarked", axis=1)
>>> df = df.drop("Cabin", axis=1)
>>> df = df.drop("Name", axis=1)
>>> df
     PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch  \
0              1         0       3    male  22.0      1      0
1              2         1       1  female  38.0      1      0
2              3         1       3  female  26.0      0      0
3              4         1       1  female  35.0      1      0
4              5         0       3    male  35.0      0      0
5              6         0       3    male   NaN      0      0
6              7         0       1    male  54.0      0      0
7              8         0       3    male   2.0      3      1
8              9         1       3  female  27.0      0      2
9             10         1       2  female  14.0      1      0
10            11         1       3  female   4.0      1      1
11            12         1       1  female  58.0      0      0
12            13         0       3    male  20.0      0      0
13            14         0       3    male  39.0      1      5
14            15         0       3  female  14.0      0      0
15            16         1       2  female  55.0      0      0
16            17         0       3    male   2.0      4      1
17            18         1       2    male   NaN      0      0
18            19         0       3  female  31.0      1      0
19            20         1       3  female   NaN      0      0
20            21         0       2    male  35.0      0      0
21            22         1       2    male  34.0      0      0
22            23         1       3  female  15.0      0      0
23            24         1       1    male  28.0      0      0
24            25         0       3  female   8.0      3      1
25            26         1       3  female  38.0      1      5
26            27         0       3    male   NaN      0      0
27            28         0       1    male  19.0      3      2
28            29         1       3  female   NaN      0      0
29            30         0       3    male   NaN      0      0
..           ...       ...     ...     ...   ...    ...    ...
861          862         0       2    male  21.0      1      0
862          863         1       1  female  48.0      0      0
863          864         0       3  female   NaN      8      2
864          865         0       2    male  24.0      0      0
865          866         1       2  female  42.0      0      0
866          867         1       2  female  27.0      1      0
867          868         0       1    male  31.0      0      0
868          869         0       3    male   NaN      0      0
869          870         1       3    male   4.0      1      1
870          871         0       3    male  26.0      0      0
871          872         1       1  female  47.0      1      1
872          873         0       1    male  33.0      0      0
873          874         0       3    male  47.0      0      0
874          875         1       2  female  28.0      1      0
875          876         1       3  female  15.0      0      0
876          877         0       3    male  20.0      0      0
877          878         0       3    male  19.0      0      0
878          879         0       3    male   NaN      0      0
879          880         1       1  female  56.0      0      1
880          881         1       2  female  25.0      0      1
881          882         0       3    male  33.0      0      0
882          883         0       3  female  22.0      0      0
883          884         0       2    male  28.0      0      0
884          885         0       3    male  25.0      0      0
885          886         0       3  female  39.0      0      5
886          887         0       2    male  27.0      0      0
887          888         1       1  female  19.0      0      0
888          889         0       3  female   NaN      1      2
889          890         1       1    male  26.0      0      0
890          891         0       3    male  32.0      0      0

               Ticket      Fare
0           A/5 21171    7.2500
1            PC 17599   71.2833
2    STON/O2. 3101282    7.9250
3              113803   53.1000
4              373450    8.0500
5              330877    8.4583
6               17463   51.8625
7              349909   21.0750
8              347742   11.1333
9              237736   30.0708
10            PP 9549   16.7000
11             113783   26.5500
12          A/5. 2151    8.0500
13             347082   31.2750
14             350406    7.8542
15             248706   16.0000
16             382652   29.1250
17             244373   13.0000
18             345763   18.0000
19               2649    7.2250
20             239865   26.0000
21             248698   13.0000
22             330923    8.0292
23             113788   35.5000
24             349909   21.0750
25             347077   31.3875
26               2631    7.2250
27              19950  263.0000
28             330959    7.8792
29             349216    7.8958
..                ...       ...
861             28134   11.5000
862             17466   25.9292
863          CA. 2343   69.5500
864            233866   13.0000
865            236852   13.0000
866     SC/PARIS 2149   13.8583
867          PC 17590   50.4958
868            345777    9.5000
869            347742   11.1333
870            349248    7.8958
871             11751   52.5542
872               695    5.0000
873            345765    9.0000
874         P/PP 3381   24.0000
875              2667    7.2250
876              7534    9.8458
877            349212    7.8958
878            349217    7.8958
879             11767   83.1583
880            230433   26.0000
881            349257    7.8958
882              7552   10.5167
883  C.A./SOTON 34068   10.5000
884   SOTON/OQ 392076    7.0500
885            382652   29.1250
886            211536   13.0000
887            112053   30.0000
888        W./C. 6607   23.4500
889            111369   30.0000
890            370376    7.7500

[891 rows x 9 columns]
>>> df = df.drop("Ticket", axis=1)
>>> df
     PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch      Fare
0              1         0       3    male  22.0      1      0    7.2500
1              2         1       1  female  38.0      1      0   71.2833
2              3         1       3  female  26.0      0      0    7.9250
3              4         1       1  female  35.0      1      0   53.1000
4              5         0       3    male  35.0      0      0    8.0500
5              6         0       3    male   NaN      0      0    8.4583
6              7         0       1    male  54.0      0      0   51.8625
7              8         0       3    male   2.0      3      1   21.0750
8              9         1       3  female  27.0      0      2   11.1333
9             10         1       2  female  14.0      1      0   30.0708
10            11         1       3  female   4.0      1      1   16.7000
11            12         1       1  female  58.0      0      0   26.5500
12            13         0       3    male  20.0      0      0    8.0500
13            14         0       3    male  39.0      1      5   31.2750
14            15         0       3  female  14.0      0      0    7.8542
15            16         1       2  female  55.0      0      0   16.0000
16            17         0       3    male   2.0      4      1   29.1250
17            18         1       2    male   NaN      0      0   13.0000
18            19         0       3  female  31.0      1      0   18.0000
19            20         1       3  female   NaN      0      0    7.2250
20            21         0       2    male  35.0      0      0   26.0000
21            22         1       2    male  34.0      0      0   13.0000
22            23         1       3  female  15.0      0      0    8.0292
23            24         1       1    male  28.0      0      0   35.5000
24            25         0       3  female   8.0      3      1   21.0750
25            26         1       3  female  38.0      1      5   31.3875
26            27         0       3    male   NaN      0      0    7.2250
27            28         0       1    male  19.0      3      2  263.0000
28            29         1       3  female   NaN      0      0    7.8792
29            30         0       3    male   NaN      0      0    7.8958
..           ...       ...     ...     ...   ...    ...    ...       ...
861          862         0       2    male  21.0      1      0   11.5000
862          863         1       1  female  48.0      0      0   25.9292
863          864         0       3  female   NaN      8      2   69.5500
864          865         0       2    male  24.0      0      0   13.0000
865          866         1       2  female  42.0      0      0   13.0000
866          867         1       2  female  27.0      1      0   13.8583
867          868         0       1    male  31.0      0      0   50.4958
868          869         0       3    male   NaN      0      0    9.5000
869          870         1       3    male   4.0      1      1   11.1333
870          871         0       3    male  26.0      0      0    7.8958
871          872         1       1  female  47.0      1      1   52.5542
872          873         0       1    male  33.0      0      0    5.0000
873          874         0       3    male  47.0      0      0    9.0000
874          875         1       2  female  28.0      1      0   24.0000
875          876         1       3  female  15.0      0      0    7.2250
876          877         0       3    male  20.0      0      0    9.8458
877          878         0       3    male  19.0      0      0    7.8958
878          879         0       3    male   NaN      0      0    7.8958
879          880         1       1  female  56.0      0      1   83.1583
880          881         1       2  female  25.0      0      1   26.0000
881          882         0       3    male  33.0      0      0    7.8958
882          883         0       3  female  22.0      0      0   10.5167
883          884         0       2    male  28.0      0      0   10.5000
884          885         0       3    male  25.0      0      0    7.0500
885          886         0       3  female  39.0      0      5   29.1250
886          887         0       2    male  27.0      0      0   13.0000
887          888         1       1  female  19.0      0      0   30.0000
888          889         0       3  female   NaN      1      2   23.4500
889          890         1       1    male  26.0      0      0   30.0000
890          891         0       3    male  32.0      0      0    7.7500

[891 rows x 8 columns]
>>> df.isnull().sum()
PassengerId      0
Survived         0
Pclass           0
Sex              0
Age            177
SibSp            0
Parch            0
Fare             0
dtype: int64
>>> df = df.dropna(how='any')
>>> df
     PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch      Fare
0              1         0       3    male  22.0      1      0    7.2500
1              2         1       1  female  38.0      1      0   71.2833
2              3         1       3  female  26.0      0      0    7.9250
3              4         1       1  female  35.0      1      0   53.1000
4              5         0       3    male  35.0      0      0    8.0500
6              7         0       1    male  54.0      0      0   51.8625
7              8         0       3    male   2.0      3      1   21.0750
8              9         1       3  female  27.0      0      2   11.1333
9             10         1       2  female  14.0      1      0   30.0708
10            11         1       3  female   4.0      1      1   16.7000
11            12         1       1  female  58.0      0      0   26.5500
12            13         0       3    male  20.0      0      0    8.0500
13            14         0       3    male  39.0      1      5   31.2750
14            15         0       3  female  14.0      0      0    7.8542
15            16         1       2  female  55.0      0      0   16.0000
16            17         0       3    male   2.0      4      1   29.1250
18            19         0       3  female  31.0      1      0   18.0000
20            21         0       2    male  35.0      0      0   26.0000
21            22         1       2    male  34.0      0      0   13.0000
22            23         1       3  female  15.0      0      0    8.0292
23            24         1       1    male  28.0      0      0   35.5000
24            25         0       3  female   8.0      3      1   21.0750
25            26         1       3  female  38.0      1      5   31.3875
27            28         0       1    male  19.0      3      2  263.0000
30            31         0       1    male  40.0      0      0   27.7208
33            34         0       2    male  66.0      0      0   10.5000
34            35         0       1    male  28.0      1      0   82.1708
35            36         0       1    male  42.0      1      0   52.0000
37            38         0       3    male  21.0      0      0    8.0500
38            39         0       3  female  18.0      2      0   18.0000
..           ...       ...     ...     ...   ...    ...    ...       ...
856          857         1       1  female  45.0      1      1  164.8667
857          858         1       1    male  51.0      0      0   26.5500
858          859         1       3  female  24.0      0      3   19.2583
860          861         0       3    male  41.0      2      0   14.1083
861          862         0       2    male  21.0      1      0   11.5000
862          863         1       1  female  48.0      0      0   25.9292
864          865         0       2    male  24.0      0      0   13.0000
865          866         1       2  female  42.0      0      0   13.0000
866          867         1       2  female  27.0      1      0   13.8583
867          868         0       1    male  31.0      0      0   50.4958
869          870         1       3    male   4.0      1      1   11.1333
870          871         0       3    male  26.0      0      0    7.8958
871          872         1       1  female  47.0      1      1   52.5542
872          873         0       1    male  33.0      0      0    5.0000
873          874         0       3    male  47.0      0      0    9.0000
874          875         1       2  female  28.0      1      0   24.0000
875          876         1       3  female  15.0      0      0    7.2250
876          877         0       3    male  20.0      0      0    9.8458
877          878         0       3    male  19.0      0      0    7.8958
879          880         1       1  female  56.0      0      1   83.1583
880          881         1       2  female  25.0      0      1   26.0000
881          882         0       3    male  33.0      0      0    7.8958
882          883         0       3  female  22.0      0      0   10.5167
883          884         0       2    male  28.0      0      0   10.5000
884          885         0       3    male  25.0      0      0    7.0500
885          886         0       3  female  39.0      0      5   29.1250
886          887         0       2    male  27.0      0      0   13.0000
887          888         1       1  female  19.0      0      0   30.0000
889          890         1       1    male  26.0      0      0   30.0000
890          891         0       3    male  32.0      0      0    7.7500

[714 rows x 8 columns]
>>> df.to_csv("usetrain.csv", index=False)
```

