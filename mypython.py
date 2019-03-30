#!/usr/bin/python

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn import datasets

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

boston = datasets.load_boston()
df = pd.DataFrame(boston.data,columns=boston.feature_names)
df['MEDV'] = boston.target
df.head()

ax, fig = plt.subplots(figsize = (16,10))
# annot = True to print the values inside the square
sns.heatmap(df.corr(), annot= True, cmap = 'RdBu')
plt.savefig("heatmap.png")
plt.show()



data = boston.data         #sample
target = boston.target     #sample target

X_train = data[:450]
y_train = target[:450]

X_test = data[450:]
y_test = target[450:]

lr = LinearRegression()
rr = Ridge()
lasso = Lasso()

lr.fit(X_train,y_train)
rr.fit(X_train,y_train)
lasso.fit(X_train,y_train)

y_lr_ = lr.predict(X_test)
y_rr_ = rr.predict(X_test)
y_lasso_ = lasso.predict(X_test)

plt.plot(y_lr_,label='lr')
plt.plot(y_rr_,label='rr')
plt.plot(y_lasso_,label='lasso')
plt.legend()
plt.savefig("predict.png")
plt.show()


X_train = df['LSTAT']
y_train = target
plt.scatter(X_train,y_train)
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.savefig("LSTAT.png")
plt.show()

X_train = df['RM']
y_train = target
plt.scatter(X_train,y_train)
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.savefig("RM.png")
plt.show()
