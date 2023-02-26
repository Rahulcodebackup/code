import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
iris.feature_names
iris.target_names
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

df['target']=iris.target #target o means setosa
df.head()

df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
df.head()
df0=df[:50]
df1=df[50:100]
df2=df[100:]

import matplotlib.pyplot as plt
#sepal length vs sepal width ()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

from sklearn.model_selection import train_test_split
x=df.drop(['target','flower_name'],axis=1)
y=df.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
len(x_train)
len(x_test)

##Create KNN (K Nearest Neighbour Classifier)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)

y_pred=knn.predict(x_test)
y_pred

##plot confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm

import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')