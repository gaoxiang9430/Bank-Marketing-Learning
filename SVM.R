rm(list=ls())
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
train <- read.csv("~/Documents/R/data/train.csv",na.strings = c("unknown"))
prop.table(table(train$y))
originaltest <- read.csv("~/Documents/R/data/test.csv",na.strings = c("unknown"))
test <- originaltest
train$y <- ifelse(train$y == "yes", 1, 0)
test$y <- ifelse(test$y == "yes", 1, 0)
train$y <- as.factor(train$y)

#table(train$y)
#remove default column, otherwise, may trigger an error, do not the exact reason
train <- train[,-6]
test <- test[,-6]

#importance attributes from Toan
#train <- train[, c("duration", "euribor3m", "age", "nr.employed", "job", "day_of_week", "education", "campaign", "pdays", "month", "poutcome", "y")]
#test <- test[, c("duration", "euribor3m", "age", "nr.employed", "job", "day_of_week", "education", "campaign", "pdays", "month", "poutcome")]

#train <- train[,c("duration", "euribor3m", "age", "nr.employed", "job", "day_of_week", "education", "campaign", "pdays", "poutcome", "month", "cons.conf.idx", "cons.price.idx","emp.var.rate","y")] 
#test <- test[,c("duration", "euribor3m", "age", "nr.employed", "job", "day_of_week", "education", "campaign", "pdays", "poutcome", "month", "cons.conf.idx", "cons.price.idx","emp.var.rate")] 

#remove class column
setDT(train)
setDT(test)

#deal with data missing
table(is.na(train))
table(is.na(test))
trainTemp <- data.frame(train)
testTemp <- data.frame(test)
#imp1 <- impute(trainTemp,classes = list(integer=imputeLearner("classif.rpart"), factor=imputeLearner("classif.rpart")))
#imp2 <- impute(testTemp,classes = list(integer=imputeLearner("classif.rpart"), factor=imputeLearner("classif.rpart")))
imp1 <- mlr::impute(trainTemp,classes = list(integer=imputeMedian(), factor=imputeMode()))
imp2 <- mlr::impute(testTemp,classes = list(integer=imputeMedian(), factor=imputeMode()))
train <- imp1$data
test <- imp2$data

#if the value is character, make it as factor
fact_col <- colnames(train)[sapply(train,is.character)]
for(i in fact_col)
  set(train,j=i,value = factor(train[[i]]))

fact_col <- colnames(test)[sapply(test,is.character)]
for(i in fact_col)
  set(test,j=i,value = factor(test[[i]]))

#select 3/4 data as training data, the remaining part as testing data. For local testing
set.seed(754)
TrainingDataIndex <- createDataPartition(train$y, p=0.8, list = FALSE)
trainFromTrainingSet <- train[TrainingDataIndex,]
testFromTrainingSet <- train[-TrainingDataIndex,]

#listFilterMethods()
#create a learner
#generate suitable parameters
#getParamSet(rf.lrn)
#rf.lrn$par.vals <- list(ntree = 100L, importance=TRUE, cutoff = c(0.75,0.25))
#params <- makeParamSet(makeIntegerParam("mtry",lower = 2,upper = 10),makeIntegerParam("nodesize",lower = 10,upper = 50), makeIntegerParam("ntree", lower = 500, upper = 2000))
#ctrl <- makeTuneControlRandom(maxit = 10L)
#tune <- tuneParams(learner = rf.lrn, task = allTraintask, resampling = rdesc, measures = list(mcc), par.set = params, control = ctrl, show.info = T)

#set most promissing parameters
#install.packages('ROSE')
#library('ROSE')

#build two task for training and testing
#install.packages('FSelector')
#library('FSelector')

svm_model <- svm(y ~ ., trainFromTrainingSet, cost=10, kernel="polynomial", degree=3)
pred <- predict(svm_model, newdata = testFromTrainingSet)
accuracy.meas(testFromTrainingSet$y, pred)
roc.curve(testFromTrainingSet$y, pred, plotit = F)

svm_model <- svm(y ~ ., train, cost=10, kernel="polynomial", degree=3)
pred <- predict(svm_model, newdata = test)
solution <- data.frame(id = originaltest$id, prediction = as.data.frame(pred)$response)
#solution$prediction <- ifelse(solution$prediction == "yes", 1, 0)
write.csv(solution, file = '~/Documents/R/data/svm_12.csv', row.names = F)



