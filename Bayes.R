#remove environment
rm(list=ls())
# Load packages
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

test$age = cut(test$age, 10)
#rainFromTrainingSet$pdays = cut(trainFromTrainingSet$pdays, 10)
test$duration = cut(test$duration, 60)
test$euribor3m = cut(test$euribor3m, 20)

TrainingDataIndex <- createDataPartition(train$y, p=0.75, list = FALSE)
trainFromTrainingSet <- train[TrainingDataIndex,]
testFromTrainingSet <- train[-TrainingDataIndex,]

#training Native bayes model
NBModel <- train(train[,c(-1,-22)],as.factor(train$y), method = "nb",trControl= trainControl(method = "cv", number = 10, repeats = 5),preProcess = c("pca"),na.action = na.omit)
#NBModel <- naiveBayes(factor(y) ~ age 
#                        + duration + pdays + emp.var.rate
#                        + cons.price.idx + cons.conf.idx + euribor3m + nr.employed,
#                        data = trainFromTrainingSet)

NBModel

#predict based on the NBModel
NBPredictions <-predict(NBModel, test, na.action = na.pass)

solution <- data.frame(id = test$id, prediction = NBPredictions)
write.csv(solution, file = 'data/nb_mod_Solution.csv', row.names = F)

