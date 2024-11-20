# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: vishnu .k m
RegisterNumber:212223240185
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/74bb56a2-4435-47d2-abdb-b4888ee9263b)
data.head():
![image](https://github.com/user-attachments/assets/b10953ba-a8e7-47f4-ace4-866ef48b03e6)
data.info():
![image](https://github.com/user-attachments/assets/f175d71b-6f9a-49f1-8fdb-2dd076094576)
data.isnull().sum():
![image](https://github.com/user-attachments/assets/0c6a7bc1-adc9-48ba-acfd-775fc29d11d7)
Y_prediction value:
![image](https://github.com/user-attachments/assets/d303ec31-fe9e-44fa-8d74-31fa616a63ac)
Accuracy value:
![image](https://github.com/user-attachments/assets/960260eb-cabf-434e-b786-b988506a2427)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
