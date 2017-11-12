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
originaltest <- read.csv("data/test.csv",na.strings = c("unknown"))

test <- originaltest

train$y <- ifelse(train$y == "yes", 1, 0)
test$y <- ifelse(test$y == "yes", 1, 0)
train$y <- as.factor(train$y)

#remove default column, otherwise, may trigger an error, do not the exact reason
train <- train[,-6]
test <- test[,-6]

#importance attributes from Toan
train <- train[,c("age", "job", "education", "month", "day_of_week", "duration", "campaign", "pdays", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y")]
test <- test[,c("age", "job", "education", "month", "day_of_week", "duration", "campaign", "pdays", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")]
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
imp1 <- impute(trainTemp,classes = list(integer=imputeMedian(), factor=imputeMode()))
imp2 <- impute(testTemp,classes = list(integer=imputeMedian(), factor=imputeMode()))
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
TrainingDataIndex <- createDataPartition(train$y, p=0.75, list = FALSE)
trainFromTrainingSet <- train[TrainingDataIndex,]
testFromTrainingSet <- train[-TrainingDataIndex,]

#build two tasks for local testing
traintask <- makeClassifTask(data = trainFromTrainingSet,target = "y")
traintask <- filterFeatures(traintask, method = "rf.importance", abs = 8)

#build two task for training and testing
allTraintask <- makeClassifTask(data = train,target = "y")
allTraintask <- filterFeatures(allTraintask, method = "rf.importance", abs = 8)

#create a learner
rdesc <- makeResampleDesc("CV",iters=5L)
rf.lrn <- makeLearner("classif.randomForest")

#generate suitable parameters
#getParamSet(rf.lrn)
#rf.lrn$par.vals <- list(ntree = 100L, importance=TRUE, cutoff = c(0.75,0.25))
#params <- makeParamSet(makeIntegerParam("mtry",lower = 2,upper = 10),makeIntegerParam("nodesize",lower = 10,upper = 50), makeIntegerParam("ntree", lower = 500, upper = 2000))
#ctrl <- makeTuneControlRandom(maxit = 10L)
#tune <- tuneParams(learner = rf.lrn, task = allTraintask, resampling = rdesc, measures = list(mcc), par.set = params, control = ctrl, show.info = T)

#set most promissing parameters
control <- trainControl(method="repeatedcv", number=10, repeats=5)
rf.lrn$par.vals <- list(importance=TRUE, ntree = 603, mtry = 5, nodesize=36, cutoff = c(0.8455,0.1545), preProcess = c("pca"), trControl=control)
#r <- resample(learner = rf.lrn, task = allTraintask, resampling = rdesc, measures = list(tpr,fpr,fnr,fpr,acc, mcc), show.info = T)

#local training and evaluation
#model <- train(rf.lrn, traintask)
#pred <- predict(model, newdata = testFromTrainingSet)
#calculateConfusionMatrix(pred)
#mcc(as.data.frame(pred)$response, as.data.frame(pred)$truth)
#d = generateThreshVsPerfData(pred, measures = list(mcc))
#plotThreshVsPerf(d)

#training and testing
model <- train(rf.lrn, allTraintask)
pred <- predict(model, newdata = test)

#write results back to file
solution <- data.frame(id = originaltest$id, prediction = as.data.frame(pred)$response)
write.csv(solution, file = 'data/rf_mod_Solution3_2.csv', row.names = F)
