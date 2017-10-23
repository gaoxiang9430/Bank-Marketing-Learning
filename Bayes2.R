#remove environment
rm(list=ls())
# Load packages
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library('caret')
library('e1071')

#you need to change to your files locations
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")
train$y <- ifelse(train$y == "yes", 1, 0)
test$y <- ifelse(test$y == "yes", 1, 0)
sum(is.na(train))
summary(train)


# Set a random seed
set.seed(754)

train$age = cut(train$age, 10)
#rainFromTrainingSet$pdays = cut(trainFromTrainingSet$pdays, 10)
train$duration = cut(train$duration, 60)
train$euribor3m = cut(train$euribor3m, 20)

TrainingDataIndex <- createDataPartition(train$y, p=0.75, list = FALSE)
trainFromTrainingSet <- train[TrainingDataIndex,]
testFromTrainingSet <- train[-TrainingDataIndex,]

#training Native bayes model
#NBModel <- naiveBayes(trainFromTrainingSet[,-22][,-1], as.factor(trainFromTrainingSet$y))
#NBModel <- train(trainFromTrainingSet[,c(2,12,14,15,17,18,19,20,21)],as.factor(trainFromTrainingSet$y), method = "nb",trControl= trainControl(method = "cv", number = 10, repeats = 5),preProcess = c("pca"),na.action = na.omit)
NBModel <- train(trainFromTrainingSet[,c(-1,-22)],as.factor(trainFromTrainingSet$y), method = "nb",trControl= trainControl(method = "cv", number = 10, repeats = 5),preProcess = c("pca"),na.action = na.omit)
#NBModel <- naiveBayes(factor(y) ~ age 
#                        + duration + pdays + emp.var.rate
#                        + cons.price.idx + cons.conf.idx + euribor3m + nr.employed,
#                        data = trainFromTrainingSet)

NBModel

#predict based on the NBModel
NBPredictions <-predict(NBModel, testFromTrainingSet, na.action = na.pass)
confusionMatrix(NBPredictions, testFromTrainingSet$y)

