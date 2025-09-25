# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:
Import the standard libraries such as pandas module to read the corresponding csv file.

Step 2:
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

Step 3:
Import LabelEncoder and encode the corresponding dataset values.

Step 4:
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

Step 5:
Predict the values of array using the variable y_pred.

Step 6:
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

Step 7:
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

Step 8:
End the program.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AADHITHYAA L
RegisterNumber:  212224220003
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
print("Name: AADHITHYAA L")
print("Register No: 212224220003")
data1.head()

print("Name: AADHITHYAA L")
print("Register No: 212224220003")
data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print("Name: AADHITHYAA L")
print("Register No: 212224220003")
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
HEAD OF THE DATA:
<img width="1286" height="283" alt="Screenshot 2025-09-25 082531" src="https://github.com/user-attachments/assets/9d44ba14-85d3-41ed-b522-70cce63e01fa" />
SUM OD ISNULL DATA:
<img width="977" height="393" alt="Screenshot 2025-09-25 082538" src="https://github.com/user-attachments/assets/7c5fa6b6-4497-4f00-9d69-535afb640a69" />
SUM OF DUPLICATE DATA:
<img width="705" height="81" alt="Screenshot 2025-09-25 082543" src="https://github.com/user-attachments/assets/b9a43158-2ab2-423f-aeb5-b77961d66254" />
LABEL ENCODER OD DATA:

<img width="1190" height="440" alt="Screenshot 2025-09-25 083537" src="https://github.com/user-attachments/assets/a3966ace-ec3a-4ecf-be25-e9872b3ff385" />
ILOC OF X:
<img width="1316" height="446" alt="Screenshot 2025-09-25 083545" src="https://github.com/user-attachments/assets/823967a3-afb8-4b0d-964d-2a16a52fb3e7" />

<img width="735" height="265" alt="Screenshot 2025-09-25 083551" src="https://github.com/user-attachments/assets/b4a4b191-34bb-47bd-9eb8-5064e1313ea8" />
PREDICTED VALUES:
<img width="890" height="166" alt="Screenshot 2025-09-25 083558" src="https://github.com/user-attachments/assets/3f154964-5538-4939-bf17-3398527ea3ba" />
ACCURACY:
<img width="521" height="70" alt="Screenshot 2025-09-25 083602" src="https://github.com/user-attachments/assets/31608925-5e80-4ff0-9396-e31a907d6d81" />
CONFUSION MATRIX:
<img width="607" height="65" alt="Screenshot 2025-09-25 083607" src="https://github.com/user-attachments/assets/7b76b4cb-5481-431b-bbb7-c7d53f87243a" />
CLASSIFICATION REPORT:
<img width="607" height="65" alt="Screenshot 2025-09-25 083607" src="https://github.com/user-attachments/assets/8fc5ccb0-32e8-4127-98ce-439e7deca094" />
PREDICTED LR VALUE:
<img width="1395" height="105" alt="Screenshot 2025-09-25 083621" src="https://github.com/user-attachments/assets/4f9ca4d7-6dd8-4724-b85c-37e722f9b944" />
<img width="463" height="59" alt="Screenshot 2025-09-25 083630" src="https://github.com/user-attachments/assets/ef238855-69dd-4978-b785-4d4dcf3b61f7" />















## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
