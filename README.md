# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages
2. Import the dataset to operate on
3. Split the dataset
4. Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RENUSRI NARAHARASHETTY
RegisterNumber:  212223240139
*/
```
```
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
# RESULT OUTPUT:
![image](https://github.com/Renusri-Naraharasetty/Implementation-of-SVM-For-Spam-Mail-Detection/assets/146916363/7c27b9ce-1690-4a51-b10c-0c0603a57b6e)

# data.head():
![image](https://github.com/Renusri-Naraharasetty/Implementation-of-SVM-For-Spam-Mail-Detection/assets/146916363/d861f6d6-74d1-448c-a793-ec9db34d76ce)

# data.info():
![image](https://github.com/Renusri-Naraharasetty/Implementation-of-SVM-For-Spam-Mail-Detection/assets/146916363/36af07c6-ae8a-4c5c-97ff-5c2556ea670c)

# Y_prediction value:
![image](https://github.com/Renusri-Naraharasetty/Implementation-of-SVM-For-Spam-Mail-Detection/assets/146916363/5ff0772a-3d56-46dc-9e48-ba136ad0ad3b)

# Accuracy value:
![image](https://github.com/Renusri-Naraharasetty/Implementation-of-SVM-For-Spam-Mail-Detection/assets/146916363/9d08aead-917f-45fa-a7db-ea8c925e8e25)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
