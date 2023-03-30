import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\omer\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.csv")

data =df.copy()
data= data.drop(columns="customerID",axis=1)#customer ıd sütununu kaldırdık
#print(data.info())

data=data.rename({"Partner":"marital"},axis=1)
#print(data.info())

data["TotalCharges"]=pd.to_numeric(data["TotalCharges"],errors="coerce")

data= data.dropna()

print(data.isnull().sum())

#print(data.head(50))

# plt.boxplot(data["tenure"])
# plt.show()

le=LabelEncoder()

var= data.select_dtypes(include="object").columns
#print(data.select_dtypes(include="object").columns)

data=(data[var].apply(le.fit_transform))



y=data["Churn"]
X=data.drop(columns="Churn",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42 )

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

clf= LazyClassifier()

models,predict=clf.fit(X_train,X_test,y_train,y_test)
print(models.sort_values(by="Accuracy",ascending=False))



