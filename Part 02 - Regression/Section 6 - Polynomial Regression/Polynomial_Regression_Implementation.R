# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# # Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling

# !We won't need feature scaling since a polynomial regression model is a 
#  a multiple linear regression model with polynomial terms. !!!!!

####-----------------------------------------------------------------------###
####                  Fitting two models to our data                       ###
####-----------------------------------------------------------------------###


#Fitting Linear Regression Model.
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

#Fitting Polynomial Regression Model.
#We create an additional set of features that we will input to our multiple linear
#regression model. 
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula = Salary ~ .,
              data = dataset) 

summary(poly_reg)


####-----------------------------------------------------------------------###
####                  Visualizing the Models - Linear Regression Model     ###
####-----------------------------------------------------------------------###

ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary),
                      colour = 'red') +
geom_line(aes(x=dataset$Level , y=predict(lin_reg, new_data=dataset)),
          colour = 'blue') + 
          ggtitle('Truth or Bluff(Linear Regression)') +
          xlab('Levels') +
          ylab('Salary')

####-----------------------------------------------------------------------###
####                  Visualizing the Models - Polynomial Model            ###
####-----------------------------------------------------------------------###

ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary),
                      colour = 'red') +
  geom_line(aes(x=dataset$Level, y=predict(poly_reg, new_data=dataset)),
            colour = 'blue') + 
  ggtitle('Truth or Bluff(Polynomial Model)') +
  xlab('Levels') +
  ylab('Salary')


####-----------------------------------------------------------------------###
####                  Predicting a new result - Linear Regression          ###
####-----------------------------------------------------------------------###
y_pred = predict(lin_reg, data.frame(Level=6.5))

####-----------------------------------------------------------------------###
####                  Predicting a new result - Polynomial Model           ###
####-----------------------------------------------------------------------###

y_pred = predict(poly_reg, data.frame(Level=6.5),
                                      Level2=6.5^2,
                                      Level3=6.5^3,
                                      Level4=6.5^4))


                                      





