import os
import pandas as pd
import matplotlib.pyplot as plt
os.chdir("D:/Praveen Naidu/Data Science/Python/Day10 KNN ")
df=pd.read_csv("Telecustomers.csv")
df.columns
x=df.drop(['custcat'],axis=1)
y=df['custcat']

from sklearn import preprocessing
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x
y
# Dataset

# As you can see, there are 12 columns, 
# namely as region, tenure, age, marital, address, income, ed, employ, 
# retire, gender, reside, and custcat. 
# We have a target column, ‘custcat’ categorizes the customers into four groups:

# 1- Basic Service
# 2- E-Service
# 3- Plus Service
# 4- Total Service

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
x_train.shape

k=5
from sklearn.neighbors import KNeighborsClassifier
neigh=KNeighborsClassifier(n_neighbors=k)
neigh.fit(x_train,y_train)
neigh.score(x_test,y_test)

import numpy as np
error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(x_train,y_train)
 pred_i = knn.predict(x_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

#From the plot, you can see that the smallest error we got is 0.59 at K=37.
#Further on, we visualize the plot between accuracy and K value.

acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
    yhat = neigh.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, yhat)
               
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

# Now you see the improved results. 
# We got the accuracy of 0.41 at K=37.
# As we already derived the error plot and got the minimum error at k=37,
# so we will get better efficiency at that K value.