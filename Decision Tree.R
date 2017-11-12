## load data into R
mydata=read.csv("train.csv")
summary(mydata)

#histogram
library(ggplot2)
ggplot(mydata,aes(x=age)) + geom_histogram(aes(y=..density..))
#box plot
boxplot(mydata$euribor3m)

## remove coumn "id"
mydata$id <- NULL
head(mydata) 

## replace the missing value by median for numeric columns and the most frequent value (the mode) for nominal variables
library(plyr)
junkframe<-mydata
count(junkframe, 'marital')
junkframe$marital[junkframe$marital == "unknown"] <- "married"
count(junkframe, 'default')
junkframe$marital[junkframe$marital == "unknown"] <- "no"
count(junkframe, 'housing')
junkframe$marital[junkframe$marital == "unknown"] <- "yes"
count(junkframe, 'loan')
junkframe$marital[junkframe$marital == "unknown"] <- "no"


## build model
library(rpart)
# grow tree 
fit <- rpart(y ~ .,method="class", data=junkframe)
printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits
# plot tree 
plot(fit, uniform=TRUE,main="Classification Tree for y")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

##predicting using the model
testdata = read.csv("test.csv") 
pred <- predict(fit, newdata = testdata)

##add id and reduce to y
colnames(pred) <- c("id", "prediction")
predframe = data.frame(pred)
predframe$prediction[predframe$prediction >= 0.5] <-1
predframe$prediction[predframe$prediction < 0.5] <- 0
predframe$id<-0:(nrow(predframe)-1)

##save
write.csv(predframe,"ab.csv")