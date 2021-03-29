# Libraries Needed
install.packages("caret")
library(caret)
install.packages("glmnet")
library(glmnet)
install.packages("mlbench")
library(mlbench)
install.packages("psych")
library(psych)

# Data
data("BostonHousing")
data <- BostonHousing
str(data)

#Lets look at correlation between independent variables
#BostonHousing is part of mlbench
#pairs.panel is part of psych package
#out of all these variables, i want to exclude 'chas, at 4th location' and 'medv, at 14th location'
pairs.panels(data[c(-4, -14)], cex = 2)
#somehow this line isnt working
#this gives us a scatter plot for every possible combination of independent variables
#when numeric independent variables are highly correlated..
  #that created multi coleanirty problem
  #and when we do multiple linear regression, 
  #the estimates we get are not very stable
  #because of that the prediction model we are trying to develop..
  #may not be very accurate
#Problem:
  #Colleanirty leads to overfitting
#Solution
#1. Ridge Regression:
  #shrinks coefficients to non zero values to prevent overfit, but keeps
    #all variables
#2. Lasso Regression:
  #shrinks regression coefficients, with some shrunk to zero. Thus it also ..
  #helps with feature selection
#3. Elastic Net Regression:
  #Mix of Ridge and Lasso

#Someofsquares for Ridge = total of actual response value - predicted response value and whole square
#in multiple linear regression, we try to minimize this above error, when developing the model
#extra component in ridge regression is l2, which is called penalty term
#have taken a screen shot of the penalty term

#while minimizing sum of squares due to errors using ridge reg
  #the penalty term makes coefficients to shrink

#when you look at sumofsquares of lasso = 
  #...whole square + lambda times sum of absoulte values of beta
  #which is also called l1 penalty term
  #this l1, not only shrinks, but shrinks some of them to zero
  #that is very useful for feature selection

#Using these two above, we can write, 
  #Sum of squares for elasticnet: taken the screenshot for eqn
  #when alpha is 0, the last part of alpha will vanish
  #and when alpha is 1, the first part of alpha will vanish
# Therefore when alpha is 0, the elasticnet reduces to ridge
# when alpha is 1, the elasticnet reduces to lasso
#Elasticnet Regression models are more flexible
  #so when we fit elasticnet models,
  #we may end up with best model
  #as 20% ridge
  #and remanining 80% as lasso
  #or it could be some other combination of ridge and lasso





# Data Partition
set.seed(222)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
#we are taking two independent samples, with replacement

train <- data[ind==1,]#70% goes to train
test <- data[ind==2,]

# Custom Control Parameters
#in 10 fold cross validation, training data is broken into 10 parts
#and then model is made from 9 parts and 1 part is used for error estimation
#this is repeated 10 times with a different part used for error estimation
custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter = T)#when program is running, we see whats going on

# Multiple Linear Model
set.seed(1234)
lm <- train(medv ~ .,
            train,
            method = 'lm',
            trControl = custom)

# Results
lm$finalModel
#above code tells us:
  #it tells us we have a model with intercept
  #we have rmse
  #we have rsquare: coefficient of determination
    #which is 0.78: meaning, more than 78% of variablity seen in response.
    # which is  medv is bse of the model
  #MAE: should always be lower
  #and others as shown

lm
#above code
#it says that its a linear regression
  #we have used 353 samples (in our training data we have had 353)
  #the training data is divided into 10 parts
  #and each time it uses 9 parts to create the model
  #and the 10th part is used for testing the error
  #and this is repeated 5 times
#RMSE is based on data that was not used while creating the model
  #as we see summary of sample size: 317, 317
  #out of 353, we have 317, so remaining data points were used to create RMSE

summary(lm)
#here those variables that do not have stars are statistically significant

plot(lm$finalModel)
#ideally all the points should fall on straight line





# Ridge Regression
#it tries to shrink coefficients but keeps the variables in the model
set.seed(1234)
ridge <- train(medv ~ .,
               train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001, 1, length = 5)),
               trControl = custom) #this is our custom control

#after it runs above, it finds the best value of lambda is 0.5
#lambda is hyperparameter, is estimated using cross validation
  #it is basically strenght of penalty on the coefficients

#as we increase lambda, we increase the penalty
#as we decrease lambda, we are reducing the penalty
#so when lambda is increased it makes the coefficents to shrink

# ------------------------------------------------------------------------------
# Plot Results
plot(ridge)
#on y axis we have RMSE, which is calculated based on repeated CV
#we can see, for higher values of lambda, error increases
#we can see the best value of lambda is 0.5

ridge
#this gives some information of the model
#it shows alpha is 0
#and best lambda value is 0.5005

plot(ridge$finalModel, xvar = "lambda", label = T)
#we see if the log lambda is around 9 or 10, all the coefficents are 0
#and as we relax lambda, coefficients start to grow
#as coefficients start to grow, sum of squares of coefficents likely to become larger and larger
#at top it says that we have all 13 independent variables in the model
#increasing lambda helps to reduce size of coefficients
  #and it makes those variables which are not contributing to 0

plot(ridge$finalModel, xvar = 'dev', label=T)
#this gives fraction deviance explained
#we see 20% variabiluty is explained by that point touching to middle line
  #with slight growth in coefficents
#by the time we reach at 60% of deviance, there is suddne jump (the blue line)
  #the coefficent becomes highly inflated
  #that means in this area, we are doing more of over fitting

plot(varImp(ridge, scale=T))
#it shows that nox is most important variable

# Lasso Regression 
#Least Absolute shrinkage & selection operator
#it does both shrinkage as well as feature selection
#and if there is group of highly correlated variables, which are causing multi colieanirty
#lasso then selects one variable and will ignore others

set.seed(1234)
lasso <- train(medv ~ .,
               train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 1,
                                      lambda = seq(0.0001, 1, length = 5)),
               trControl = custom)
#it gives us lambda value as small value

# Plot Results
plot(lasso)
#this shows that higher values increase RMSE
#so the optimal is towards the lower side
  #the starting point of the line is the lambda value

lasso
#above gives the output for each cross validation

plot(lasso$finalModel, xvar = 'lambda', label=T)
#label = T, to show the labels, whcih indicates that which variable is plotted
#the blue line, coefficent is growing much rapidly as we reduce lambda
#as compared to the red line (the top one)
  #which is performing better than the blue one

#Variable Importance Plot
plot(varImp(lasso, scale = T))

# Elastic Net Regression
set.seed(1234)
en <- train(medv ~., 
            train,
            method = 'glmnet',
            tuneGrid = expand.grid(alpha = seq(0,1, length = 10),
                                   lambda = seq(0.0001, 1, length = 5)),
            trControl = custom)

#alpha value = 0.111

# Plot Results
plot(en)
#for yellow line, the rmse is high for that

plot(en$finalModel, xvar = 'lambda', label=T)
#we see when loglambda is 4, coeff are almost 0
  #and the model has only two variables
  #when loglambda is 2, some of the coeff grow and have 10 variables in the model
  #when loglambda is 0, the blue line, coeff gorws rapidly
    #the other two above are still rising but not high rate as the blue one
  #the top line will have larger importance in the model
    #as compared to the blue line model

plot(en$finalModel, xvar = 'dev', label=T)

plot(varImp(en))

# Compare Models
model_list <- list(linearModel = lm, Ridge = ridge, Lasso = lasso, ElasticNet = en)
res <- resamples(model_list)
summary(res)
bwplot(res)#cant see as the values are small in dataset, hence look at numeric summary 
xyplot(res, metric = 'RMSE')
#this is the scatter plot between ridge regression model and linear reg model
#so results from cross validation and repeats, they lie close to each other for these two models
  #its by coincidence that, the data we have used here not really differentiating btn ..
  #ridge and linear model in a significant way
#in other dataset, it shows that one model shows better than the other model

#the first point seen, rmse for ridge regression is smaller than what you have for linear regression model
  #bse the dot is above the dottted line
#so the dots that are above the dotted line perform better than when we have ridge regression models
#dots which are below: perform better when we have linear regression models
#so this plot compares only ridge and linear model
#but we can have different models to be plotted


# Best Model
en$bestTune
#these are the best values 
#so point 0.111 is closer to 0, meaning that final elastic model is for ridge model
  #and less for a lasso model 

best <- en$finalModel
coef(best, s = en$bestTune$lambda)
#we see highest coeff is for nox and followed by rm


# Save Final Model for Later Use
saveRDS(en, "final_model.rds")
fm <- readRDS("final_model.rds")
print(fm)

# Prediction
p1 <- predict(fm, train)
sqrt(mean((train$medv-p1)^2)) #here i am getting RMSE for training data

p2 <- predict(fm, test)
sqrt(mean((test$medv-p2)^2))
#rmse value for testing data