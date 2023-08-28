import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt



csv=pd.read_csv('Titanic-Dataset.csv')
csv.loc[csv['Sex']=='female','Sex']=0
csv.loc[csv['Sex']=='male','Sex']=1
csv.loc[csv['Embarked']=='C','Embarked']=0
csv.loc[csv['Embarked']=='Q','Embarked']=1
csv.loc[csv['Embarked']=='S','Embarked']=2


model_data=csv[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]
# print(model_data.head())
# print(model_data.isnull().sum())
model_data.loc[pd.isnull(model_data["Age"]), 'Age'] = round(model_data['Age'].mean())
model_data.loc[pd.isnull(model_data["Embarked"]), 'Embarked'] = 1


X=model_data.iloc[:,:-1]
y=model_data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(X,y)
scaler=MinMaxScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train_scaled,y_train)
pred=knn.predict(x_test_scaled)
print('Accuracu score: ',accuracy_score(y_test,pred))