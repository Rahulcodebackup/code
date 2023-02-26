import os
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
os.chdir("D:/Praveen Naidu/Data Science/Python/Day10 KNN")
df=pd.read_csv("diabetes.csv")

non_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for coloumn in non_zero:
    df[coloumn] = df[coloumn].replace(0,np.NaN)
    mean = int(df[coloumn].mean(skipna = True))
    df[coloumn] = df[coloumn].replace(np.NaN,mean)
    print(df[coloumn])

x=df.drop(['Outcome'],axis=1)
y=df['Outcome']

from sklearn import preprocessing
x=preprocessing.StandardScaler().fit(x).transform(x.astype(float))
x
y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

from sklearn.neighbors import KNeighborsClassifier

acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
    yhat = neigh.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))