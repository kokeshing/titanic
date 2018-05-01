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
```

