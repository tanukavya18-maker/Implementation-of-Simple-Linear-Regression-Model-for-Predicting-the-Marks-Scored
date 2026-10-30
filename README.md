# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kavya G
RegisterNumber:  212225040180
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read dataset
df = pd.read_csv('student_scores.csv')

print(df)
print(df.head())
print(df.tail())

# Separate input and output
x = df.iloc[:, :-1].values
print(x)

y = df.iloc[:, 1].values
print(y)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)

# Create Linear Regression model
regressor = LinearRegression()

# Train the model
regressor.fit(x_train, y_train)

# Predict values
y_pred = regressor.predict(x_test)

print(y_pred)
print(y_test)

# Graph plot for training data
plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Graph plot for testing data
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Error calculation
mse = mean_squared_error(y_test, y_pred)
print("MSE =", mse)

mae = mean_absolute_error(y_test, y_pred)
print("MAE =", mae)

rmse = np.sqrt(mse)
print("RMSE =", rmse)

```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


DATA SET



<img width="207" height="565" alt="image" src="https://github.com/user-attachments/assets/5d6d3a45-a7cd-4124-b398-f6df8c22e6c0" />



HEAD



<img width="187" height="130" alt="image" src="https://github.com/user-attachments/assets/63835633-2280-4ccf-9e85-075f3cc791a1" />



TAIL



<img width="192" height="126" alt="image" src="https://github.com/user-attachments/assets/b391dcea-ecff-42f6-a5b8-dff58a62b560" />



X and Y values



<img width="708" height="587" alt="image" src="https://github.com/user-attachments/assets/7b9e3eb3-a7fc-470c-96ef-cee36a7d2d99" />



Predication values of X and Y



<img width="702" height="60" alt="image" src="https://github.com/user-attachments/assets/18e56c42-7f94-44f9-935b-7922c11893fe" />




MSE,MAE and RMSE



<img width="262" height="67" alt="image" src="https://github.com/user-attachments/assets/54105f6e-aa59-4d77-9890-cf40de4af9c4" />



Training Set



<img width="693" height="562" alt="image" src="https://github.com/user-attachments/assets/0a1027a3-3bba-421e-9698-ade56d5c55d9" />



Testing Set



<img width="700" height="566" alt="image" src="https://github.com/user-attachments/assets/2f61fdf4-8827-4a28-8e44-c7e5a24751db" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
