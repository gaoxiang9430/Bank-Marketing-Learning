#remove environment
rm(list=ls())
# Load packages
library('randomForest') # classification algorithm
library('caret')
library('e1071')
library('mltools')

#you need to change to your files locations
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")
train$y <- ifelse(train$y == "yes", 1, 0)
test$y <- ifelse(test$y == "yes", 1, 0)
#sum(is.na(train))
#summary(train)

TrainingDataIndex <- createDataPartition(train$y, p=0.66, list = FALSE)
trainFromTrainingSet <- train[TrainingDataIndex,]
testFromTrainingSet <- train[-TrainingDataIndex,]

train <- trainFromTrainingSet

# Set a random seed
set.seed(754)

control <- trainControl(method="repeatedcv", number=10, repeats=3)
mtry <- sqrt(13)
tunegrid <- expand.grid(.mtry=1)

# Build the model (note: not all possible variables are used)
rf_model <-randomForest(factor(y) ~ age + job  + education 
                        + month + day_of_week + duration + campaign + pdays + poutcome + emp.var.rate 
                        + cons.price.idx + euribor3m + nr.employed,
                        data = train,
                        ntree = 1300,
                        mtry = 4,
                        preProcess = c("pca"),
                        trControl=control,
                        importance=TRUE)

#RFprediction <- predict(rf_model, test, na.action = na.pass)
RFprediction <- predict(rf_model, testFromTrainingSet, na.action = na.pass)
print("RF MCC: ")
mcc(RFprediction, testFromTrainingSet$y)
confusionMatrix(RFprediction, testFromTrainingSet$y)

svm_model <-svm(factor(y) ~ age + job  + education 
                        + month + day_of_week + duration + campaign + pdays + poutcome + emp.var.rate 
                        + cons.price.idx + euribor3m + nr.employed,
                        data = train,
                        preProcess = c("pca"),
                        trControl=control)
#NBprediction <- predict(nb_model, test, na.action = na.pass)
SVMprediction <- predict(svm_model, testFromTrainingSet, na.action = na.pass)
print("SVM MCC: ")
mcc(SVMprediction, testFromTrainingSet$y)
confusionMatrix(SVMprediction, testFromTrainingSet$y)

nb_model <- naiveBayes(factor(y) ~ age + job  + education 
                        + month + day_of_week + duration + campaign + pdays + poutcome + emp.var.rate 
                        + cons.price.idx + euribor3m + nr.employed,
                        data = train,
                        preProcess = c("pca"),
                        trControl=control)
#NBprediction <- predict(nb_model, test, na.action = na.pass)
NBprediction <- predict(nb_model, testFromTrainingSet, na.action = na.pass)
print("NB MCC: ")
mcc(NBprediction, testFromTrainingSet$y)
confusionMatrix(NBprediction, testFromTrainingSet$y)

combine <- SVMprediction
for (i in c(1:10502)){
  if(RFprediction[i] == 1){
    combine[i] = 1
  }
}
confusionMatrix(combine, testFromTrainingSet$y)



# Predict using the test set
#solution <- data.frame(id = test$id, prediction = NBprediction)
#write.csv(solution, file = 'data/combine.csv', row.names = F)


