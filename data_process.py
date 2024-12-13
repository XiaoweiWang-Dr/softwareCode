import csv
import numpy as np
from numpy import unique
from numpy import where
from matplotlib import pyplot
import joblib
from sklearn.model_selection import train_test_split
import os
grandfa = os.path.dirname(os.path.realpath(__file__))
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

csv_reader = csv.reader(open(r"C:\data\TempData_A-1_2024-06-13 00_00_00至2024-06-22 23_59_59.csv",encoding='utf-8'))
t = []
t1 = []
t2 = []
t3 = []
for row in csv_reader:
    try:
        t.append(float(row[5]))
        t1.append(float(row[6]))
        t2.append(float(row[7]))
        t3.append(float(row[8]))
        # print(row[5])
    except:
        pass

t = t1+t2+t3

print(len(t))
x_list = []
y_list = []
temp_list = []
for i in range(len(t)):
    temp_list.append(t[i])
    if i%10 == 0:
        if i==0:
            temp_list = []
            continue
        x_list.append(np.array(temp_list))
        y_list.append(t[i+1])
        temp_list = []
print(x_list[0])
print(x_list[1])
data_array = np.concatenate([array[np.newaxis, :] for array in x_list])
y_list = np.array(y_list).reshape(-1,1)
# print(x_list)
# print(y_list)
x_list = data_array
print(x_list.shape)
print(y_list.shape)
X_train, X_test, Y_train, Y_test = train_test_split(x_list,y_list,test_size=0.2,random_state=44)

clf = RandomForestRegressor()
rf = clf.fit(X_train,Y_train)
y_pred = rf.predict(X_test)
print("train score：",rf.score(X_train,Y_train))
print("test score：",rf.score(X_test,Y_test))
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plt.plot(range(int(len(Y_test)*0.4),int(len(Y_test)*0.45)), (Y_test[int(len(Y_test)*0.4):int(len(Y_test)*0.45)]), marker='.',label="true value")
plt.plot(range(int(len(y_pred)*0.4),int(len(y_pred)*0.45)), (y_pred[int(len(y_pred)*0.4):int(len(y_pred)*0.45)]), marker='.',label="predict value")
plt.legend()
plt.show()