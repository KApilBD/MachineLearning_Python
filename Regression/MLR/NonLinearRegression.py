import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

# download the Dataset from below link using !wget
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
    
#Create Datafreame out of the data downloaded
df = pd.read_csv("FuelConsumptionCo2.csv")
df.head()

#Creating second dataset out of DF with required feature(columns)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#plot Emission values with respect to Engine size:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel=('ENGINESIZE')
plt.ylabel=("CO2EMISSIONS")
plt.show()

#Creating train and test dataset
msk= np.random.rand(len(df)<0.8)
train = cdf[msk]
test = cdf[~msk]

#Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Multiple Regression Model
#When more than one independent variable is present, the process is called multiple linear regression
from sklearn import linear_model        #import Linear_model from sklearn library
regr = linear_model.LinearRegression()      #create linear regression function
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])     #adding multiple independent feature
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)
# The coefficients
print ('Coefficients: ', regr.coef_)
#scikit-learn uses plain Ordinary Least Squares method to solve this problem.

#Prediction
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])  # Y^ is fuction (y^= 0o + 0102x, 0 is thita, intercept and coefficients) of multiple linear feature
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])     #adding multiple independent feature
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))



# MLR model for different feature

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
print("Coefficients: ", regr.coef_)
y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))
