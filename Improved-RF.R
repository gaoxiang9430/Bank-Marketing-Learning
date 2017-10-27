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
library('mltools')
library('data.table')
library('mlr')

#read data
train <- read.csv("data/train.csv",na.strings = c("unknown"))
test <- read.csv("data/test.csv",na.strings = c("unknown"))
#
train$y <- ifelse(train$y == "yes", 1, 0)
test$y <- ifelse(test$y == "yes", 1, 0)
train$y <- as.factor(train$y)

#remove default column, otherwise, may trigger an error, do not the exact reason
train <- train[,-6]
test <- test[,-6]

setDT(train)
setDT(test)

#deal with data missing
table(is.na(train))
table(is.na(test))
trainTemp <- data.frame(train)
#remove class column
testTemp <- data.frame(test)[,-21]

imp1 <- impute(trainTemp,classes = list(integer=imputeMedian(), factor=imputeMode()))
imp2 <- impute(testTemp,classes = list(integer=imputeMedian(), factor=imputeMode()))
train <- imp1$data
test <- imp2$data

#add fake class column
test["y"] <- 0
test$y <- as.factor(test$y)

#if the value is character, make it as factor
fact_col <- colnames(train)[sapply(train,is.character)]
for(i in fact_col)
     set(train,j=i,value = factor(train[[i]]))

fact_col <- colnames(test)[sapply(test,is.character)]
for(i in fact_col)
     set(test,j=i,value = factor(test[[i]]))

#select 3/4 data as training data, the remaining part as testing data. For local testing
set.seed(754)
TrainingDataIndex <- createDataPartition(train$y, p=0.75, list = FALSE)
trainFromTrainingSet <- train[TrainingDataIndex,]
testFromTrainingSet <- train[-TrainingDataIndex,]

#build two tasks for local testing
traintask <- makeClassifTask(data = trainFromTrainingSet,target = "y")
testtask <- makeClassifTask(data = testFromTrainingSet,target = "y")

#build two task for training and testing
allTraintask <- makeClassifTask(data = train,target = "y")
allTestask <- makeClassifTask(data = test,target = "y")

#create a learner
rdesc <- makeResampleDesc("CV",iters=5L)
rf.lrn <- makeLearner("classif.randomForest")

#generate suitable parameters
#getParamSet(rf.lrn)
#rf.lrn$par.vals <- list(ntree = 100L, importance=TRUE, cutoff = c(0.75,0.25))
#params <- makeParamSet(makeIntegerParam("mtry",lower = 2,upper = 10),makeIntegerParam("nodesize",lower = 10,upper = 50), makeIntegerParam("ntree", lower = 500, upper = 2000))
#ctrl <- makeTuneControlRandom(maxit = 10L)
#tune <- tuneParams(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)

#set most promissing parameters
rf.lrn$par.vals <- list(importance=TRUE, ntree = 500, mtry = 6, nodesize=42, cutoff = c(0.75,0.25))
#r <- resample(learner = rf.lrn, task = allTraintask, resampling = rdesc, measures = list(tpr,fpr,fnr,fpr,acc, mcc), show.info = T)

#local training and evaluation
#model <- train(rf.lrn, traintask)
#pred <- predict(model, testtask)
#calculateConfusionMatrix(pred)
#mcc(as.data.frame(pred)$response, as.data.frame(pred)$truth)

#training and testing
model <- train(rf.lrn, allTraintask)
pred <- predict(model, allTestask)

#write results back to file
solution <- data.frame(id = as.data.frame(pred)$id-1, prediction = as.data.frame(pred)$response)
write.csv(solution, file = 'data/rf_mod_Solution3.csv', row.names = F)
