library(ROSE) #Helps to generate artificial data based on sampling methods and smoothed bootstrap approach

setwd("F:/OneDrive/OneDrive - National University of Singapore/Data Mining My Project/Data-Mining-1")
training_data <- read.csv(file="data/train.csv",head=TRUE,sep=",") #Load training data
testing_data <- read.csv(file="data/test.csv",head=TRUE,sep=",") #Load testing data
print("Data read....!")

#Selecting more important variables for pre-processing
training_data <- training_data[,c("age", "job", "education", "month", "day_of_week", "duration", "campaign", "pdays", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed","y")]
print("Features selected...!")

#Addressing data inbalance
#Do both oversampling and undersampling to create a balanced dataset but can cause to lose some information and there by inaccuries
#data_balanced_both <- ovun.sample(y ~ ., data = training_data, method = "both", p=0.5, N=30891, seed = 1)$data 

#The ROSE pacakge addresses the above problem and provides balanced and informative dataset
balanced_training_data <- ROSE(y ~ ., data = training_data, seed = 1)$data
#Extract training labels
balanced_training_data <- transform(balanced_training_data, y = ifelse(y == "yes", 1, 0))
balanced_training_labels <-balanced_training_data$y
#Prints class distribution
print("Data set is now balanced...!")
table(balanced_training_data$y)
#Extract features for the training data
balanced_training_features <- subset( balanced_training_data, select = -c(y))

#Convert categorical variables for their one-hotmapping
df1 <- data.frame(model.matrix(~job-1,balanced_training_features))
df2 <- data.frame(model.matrix(~education-1,balanced_training_features))
df3 <- data.frame(model.matrix(~month-1,balanced_training_features))
df4 <- data.frame(model.matrix(~day_of_week-1,balanced_training_features))
df5 <- data.frame(model.matrix(~poutcome-1,balanced_training_features))

print("Categorical variables are converted to their one-hot mapping...!")
categorical_features_training <- cbind(df1, df2,df3,df4,df5)
continous_variables_training <- balanced_training_data[,c("age", "duration", "campaign", "pdays", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")]
final_training_data <- cbind(continous_variables_training, categorical_features_training)

#----Following code snippt is not using----------
balanced_training_features$education <- as.numeric(balanced_training_features$education)
balanced_training_features$month <- as.numeric(balanced_training_features$month)
balanced_training_features$day_of_week <- as.numeric(balanced_training_features$day_of)
balanced_training_features$job <-as.numeric(balanced_training_features$job)
balanced_training_features$poutcome <-as.numeric(balanced_training_features$poutcome)

scaled_training <- scale(final_training_data,center = TRUE,scale = TRUE)

#scaled_training <- scale(balanced_training_features,center = TRUE,scale = TRUE)

print("Data set is normalized...!")

write.csv(file="pre_processed_training_data_11_10_4.csv", x=scaled_training,row.names = FALSE) 
write.csv(file="training_labels_11_10_4.csv", x=balanced_training_labels,row.names = FALSE)

print("Preprocessed files are generated....!")

#-----------------------------------------Test data preprocessing
testing_data <- read.csv(file="data/test.csv",head=TRUE,sep=",")
testing_data <- testing_data[,c("age", "job", "education", "month", "day_of_week", "duration", "campaign", "pdays", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")]
print("Features selected...!")

#Convert categorical variables for their one-hotmapping
df1 <- data.frame(model.matrix(~job-1,testing_data))
df2 <- data.frame(model.matrix(~education-1,testing_data))
df3 <- data.frame(model.matrix(~month-1,testing_data))
df4 <- data.frame(model.matrix(~day_of_week-1,testing_data))
df5 <- data.frame(model.matrix(~poutcome-1,testing_data))

print("Categorical variables are converted to their one-hot mapping...!")
categorical_features_testing <- cbind(df1, df2,df3,df4,df5)
continous_variables_testing <- testing_data[,c("age", "duration", "campaign", "pdays", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")]
final_testing_data <- cbind(continous_variables_testing, categorical_features_testing)

#----following code snippest is not using-----
testing_data$education <- as.numeric(testing_data$education)
testing_data$month <- as.numeric(testing_data$month)
testing_data$day_of_week <- as.numeric(testing_data$day_of)
testing_data$job <-as.numeric(testing_data$job)
testing_data$poutcome <-as.numeric(testing_data$poutcome)

scaled_testing <- scale(final_testing_data,center = TRUE,scale = TRUE)
#scaled_testing <- scale(testing_data,center = TRUE,scale = TRUE)

print("Data set is normalized...!")

write.csv(file="pre_processed_testing_data_11_10_4.csv", x=scaled_testing,row.names = FALSE) 

print("Preprocessed test data file are generated....!")




