# Data Preprocessing Template

# Importing the dataset

dataset = read.csv('50_Startups.csv')
dataset$State = factor(dataset$State, 
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling won't be needed since this will be handled by the function
#that is used to fit multiple linear regression to our training set. 

regressor = lm(formula = Profit  ~ R.D.Spend + Administration
               + Marketing.Spend + State, data = training_set)
#This above statement can be written also as follows:
# regressor = lm(formula = Profit  ~ .,data=training_set)
summary(regressor)
y_pred = predict(regressor, newdata = test_set)

#Find the best optimal model for our problem with Backward Elimination. 
#that is used to fit multiple linear regression to our training set. 
#--------------------------------------------------------------------------#
#We input now the independent variables with + so when we implement the step 5 of backward
#elimination we can remove the variable that is not statistically significant. 

regressor = lm(formula = Profit  ~ R.D.Spend + Administration +  Marketing.Spend + State,
               data = dataset)
#--------------------------------------------------------------------------#
#We are taking the whole dataset so we can see which variables are statistically 
#significant and which not. 
summary(regressor)

#1 - Removing the variable that gives a p_value above the SL. 

regressor = lm(formula = Profit  ~ R.D.Spend + Administration +  Marketing.Spend,
               data = dataset)
#--------------------------------------------------------------------------#
#We are taking the whole dataset so we can see which variables are statistically 
#significant and which not. 
summary(regressor)

#2 - Removing the second variable that gives a p_value above the SL. 

regressor = lm(formula = Profit  ~ R.D.Spend + Marketing.Spend,
               data = dataset)
#--------------------------------------------------------------------------#
#We are taking the whole dataset so we can see which variables are statistically 
#significant and which not. 
summary(regressor)

#3 - Removing the second variable that gives a p_value above the SL. 
regressor = lm(formula = Profit  ~ R.D.Spend,
               data = dataset)
#--------------------------------------------------------------------------#
#We are taking the whole dataset so we can see which variables are statistically 
#significant and which not. 
summary(regressor)

#Either you eliminate the Marketing.Spend variable or keep it for your model is ok!.



