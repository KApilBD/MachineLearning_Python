import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 

#Download Dataset form the below link using !wget
!wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv

#Read the Data from csv file dataset
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5] # print top 5 data in console
#   Age	Sex	BP	Cholesterol	Na_to_K	Drug
# 0	23	F	HIGH	HIGH	25.355	drugY
# 1	47	M	LOW 	HIGH	13.093	drugC
# 2	47	M	LOW 	HIGH	10.114	drugC
# 3	28	F	NORMAL	HIGH	7.798	drugX
# 4	61	F	LOW 	HIGH	18.043	drugY

# Size of the data 
my_data.size #1200


#Pre-processing CREATE below dataset 
# X as the Feature Matrix (data of my_data) Feature
# y as the response vector (target) Label

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values # values will convert dataset into array of object type list of each row

X[0:5]
# array([[23, 'F', 'HIGH', 'HIGH', 25.355],
#        [47, 'M', 'LOW', 'HIGH', 13.093],
#        [47, 'M', 'LOW', 'HIGH', 10.113999999999999],
#        [28, 'F', 'NORMAL', 'HIGH', 7.797999999999999],
#        [61, 'F', 'LOW', 'HIGH', 18.043]], dtype=object)

#as we can see in X dataset featurs we have non numaric value"categorical values", and  Sklearn Decision Trees do not handle categorical variables 

#We will convert it using pandas.get_dummies(), Convert categorical variable into dummy/indicator variables.

from sklearn import preprocessing 
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_bp = preprocessing.LabelEncoder()
le_bp.fit(['LOW','NORMAL','HIGH'])
x[:,2]=le_bp.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]
# array([[23, 0, 0, 0, 25.355],
#        [47, 1, 1, 0, 13.093],
#        [47, 1, 1, 0, 10.113999999999999],
#        [28, 0, 2, 0, 7.797999999999999],
#        [61, 0, 1, 0, 18.043]], dtype=object)


#Same with Label Data / Response Vactor
y = my_data["Drug"]
y[0:5]
# 0    drugY
# 1    drugC
# 2    drugC
# 3    drugX
# 4    drugY
# Name: Drug, dtype: object


# Now Split the dataset into test and training

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
# Shapr of the Dataset
X_trainset.shape #(140, 5) (row, coloum)
y_testset.shape #(60,)


#Modeling.....We will first create an instance of the DecisionTreeClassifier called drugTree.

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#             splitter='best')

# fit the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)

# Prediction on the testing dataset and store
predTree = drugTree.predict(X_testset)


print (predTree [0:5])
print (y_testset [0:5])

# ['drugY' 'drugX' 'drugX' 'drugX' 'drugX']
# 40     drugY
# 51     drugX
# 139    drugX
# 197    drugX
# 170    drugX
# Name: Drug, dtype: object  It Predicted correctly and Entropy is 98333333 for it.


## Evaluation on the Dataset 
#import __metrics__ from sklearn and check the accuracy of our model.

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))    
#DecisionTrees's Accuracy:  0.9833333333333333


## Visualization.....visualize the tree

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

