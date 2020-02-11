
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

 
#We dont need feature scaling because polynomial regression consists of
#adding some polynomial terms into the multiple linear regression
#equation and therefore we will use the same l.regression library as
#the one we used when we built our linear and multiple regression 
#models. 

#Fitting linear regression model to the dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

#Plotting the linear regression results
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

#Plotting the polynomial regression
#We use fit_transform method to fit the data X to our polynomial function.
#-----------------------------------------------------#
##We create a new X matrix to create a better plot, with a higher resolution
##Instead of step=1 now we want step = 0.1

X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
#-----------------------------------------------------#
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'orange')
plt.title('Polynomial Regression Model')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

########
lin_reg.predict(6.5)
lin_reg2.predict(poly_reg.fit_transform(6.5)

########






