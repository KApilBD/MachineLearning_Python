#Importing Needed Packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

#To download the data, we will use !wget to download it from IBM Object Storage.
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
#get the FuelConsumption.csv: from the link

#Reading the data in
df = pd.read_csv("FuelConsumption.csv")

#Take a look at data
df.head() # Shows the top 5 row in Table

#Summarize the data
df.describe()

# selecting featurs and creating new DF
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9) # It will so 9 rows from top

# Plot graph for each of these feature
viz= cdf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist() #create bar graph based on time constrain
viz.show() #Show the created graph

# Plot linear graph of all feature with respect to Emission
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel('Engine Size')
plt.ylabel("Emission")
plt.show()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel("Emission")
plt.show()
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.ylabel("CYLINDERS")
plt.xlabel("EMISSIONS")
plt.show()

#Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8  #picks random record and split in 80 : 20 % ratio
train = cdf[msk]        # training data 
test = cdf[~msk]        # test data which left in mask 20 ratio

#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Modeling
# Using sklearn package to model data.

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
## The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_) #Coefficient and Intercept in the simple linear regression, are the parameters of the fit line

#plot the fit line over the data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r') # -r = red
plt.xlabel("Engine size")
plt.ylabel("Emission")

#Evaluation
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )