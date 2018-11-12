import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline

df = pd.read_csv('teleCust1000t.csv')
df.head()

#Let’s see how many of each class is in our data set
df['custcat'].value_counts()

#Historic graph for income
df.hist(column='income', bins=50)
df.columns # check all the columns

#DF Creation with required features
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

#create label 
y = df['custcat'].values
y[0:5]

#Normalize Data,,,Data Standardization give data zero mean 
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#split the datasets in test and train data
from sklearn.model_selection import train_test_split # train_test_split helps us to split the data set into test and train
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4) # test_size = 0.2 it the volume of perentage qwe want as test data
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



#K nearest neighbor (KNN)
from sklearn.neighbors import KNeighborsClassifier  

k = 4 # it coulb be any value
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train) # it takes the training data and find neighbors for each label
neigh
# #>>KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=4, p=2,
#            weights='uniform')


# Predicting the label for test data
yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy Evolution 
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# For K=6

k=6

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
neigh

yhat = neigh.predict(X_test)
yhat[0:5]

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))



#Iterated K values and its graph to check which one is most accurate 
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = []
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()