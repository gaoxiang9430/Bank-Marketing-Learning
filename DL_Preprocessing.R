library(ROSE) #Helps to generate artificial data based on sampling methods and smoothed bootstrap approach
library(plyr)


setwd("F:/OneDrive/OneDrive - National University of Singapore/Data Mining My Project/Data-Mining-1")
training_data <- read.csv(file="data/train.csv",head=TRUE,sep=",") #Load training data
testing_data <- read.csv(file="data/test.csv",head=TRUE,sep=",") #Load testing data
print("Data read....!")

#Selecting more important variables for pre-processing
training_data <- training_data[,c("age", "job", "education", "month", "day_of_week", "duration", "campaign", "pdays", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed","y")]
validation_data<- training_data[,c("age", "job", "education","month", "day_of_week", "duration", "campaign", "pdays",  "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed","y")]
testing_data <- testing_data[,c("age", "job", "education",  "month", "day_of_week", "duration", "campaign", "pdays", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")]
print("Features selected...!")

#Removing rows with "unknown" values
#training_data <-subset(training_data, education!="unknown")
#testing_data <-subset(testing_data, education!="unknown")
#validation_data <- subset(validation_data, education!="unknown")

#training_data <-subset(training_data, job!="unknown")
#testing_data <-subset(testing_data, job!="unknown")
#validation_data <- subset(validation_data, job!="unknown")

training_data$education <- revalue(training_data$education, c("unknown"="university.degree")) 
testing_data$education <-revalue(testing_data$education, c("unknown"="university.degree")) 
validation_data$education <- revalue(validation_data$education, c("unknown"="university.degree")) 

training_data$job <-revalue(training_data$job, c("unknown"="admin."))
testing_data$job <-revalue(testing_data$job, c("unknown"="admin."))
validation_data$job <- revalue(validation_data$job, c("unknown"="admin."))

#Addressing data inbalance
#Do both oversampling and undersampling to create a balanced dataset but can cause to lose some information and there by inaccuries
#data_balanced_both <- ovun.sample(y ~ ., data = training_data, method = "both", p=0.5, N=30891, seed = 1)$data 

#The ROSE pacakge addresses the above problem and provides balanced and informative dataset
balanced_training_data <- ROSE(y ~ ., data = training_data, seed = 1)$data
#balanced_training_data <- ovun.sample(y ~ ., data = training_data, method = "both", p=0.5, N=30891, seed = 1)$data


#Extract training labels
balanced_training_data <- transform(balanced_training_data, y = ifelse(y == "yes", 1, 0))
balanced_training_labels <-balanced_training_data$y

#For validation
validation_data <- transform(validation_data, y = ifelse(y == "yes", 1, 0))
validation_labels <- validation_data$y


#Prints class distribution
print("Data set is now balanced...!")
table(balanced_training_data$y)
#Extract features for the training data
balanced_training_features_ <- subset( balanced_training_data, select = -c(y))

balanced_training_features <- rbind(balanced_training_features_,testing_data)

#Convert categorical variables for their one-hotmapping
df1 <- data.frame(model.matrix(~job-1,balanced_training_features))
df2 <- data.frame(model.matrix(~education-1,balanced_training_features))
df3 <- data.frame(model.matrix(~month-1,balanced_training_features))
df4 <- data.frame(model.matrix(~day_of_week-1,balanced_training_features))
df5 <- data.frame(model.matrix(~poutcome-1,balanced_training_features))
#df6 <- data.frame(model.matrix(~housing-1,balanced_training_features))

vdf1 <- data.frame(model.matrix(~job-1,validation_data))
vdf2 <- data.frame(model.matrix(~education-1,validation_data))
vdf3 <- data.frame(model.matrix(~month-1,validation_data))
vdf4 <- data.frame(model.matrix(~day_of_week-1,validation_data))
vdf5 <- data.frame(model.matrix(~poutcome-1,validation_data))
#vdf6 <- data.frame(model.matrix(~housing-1,validation_data))

print("Categorical variables are converted to their one-hot mapping...!")
categorical_features_training <- cbind(df1, df2,df3,df4,df5)
continous_variables_training <- balanced_training_features[,c("age", "duration", "campaign", "pdays", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")]
preprocessed_data <- cbind(continous_variables_training, categorical_features_training)

#For validation dataset
v_categorical_features_training <- cbind(vdf1, vdf2,vdf3,vdf4,vdf5)
v_continous_variables_training <- validation_data[,c("age", "duration", "campaign", "pdays",  "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed")]
v_preprocessed_data <- cbind(v_continous_variables_training, v_categorical_features_training)


#----Following code snippt is not using----------
balanced_training_features$education <- as.numeric(balanced_training_features$education)
balanced_training_features$month <- as.numeric(balanced_training_features$month)
balanced_training_features$day_of_week <- as.numeric(balanced_training_features$day_of)
balanced_training_features$job <-as.numeric(balanced_training_features$job)
balanced_training_features$poutcome <-as.numeric(balanced_training_features$poutcome)

scaled_training <- scale(preprocessed_data,center = TRUE,scale = TRUE)
#scaled_training <- scale(balanced_training_features,center = TRUE,scale = TRUE)

#for validation data
scaled_validation <- scale(v_preprocessed_data,center = TRUE,scale = TRUE)

print("Data set is normalized...!")

final_training_data <- scaled_training[0:30891,]
final_testing_data <- scaled_training[30892:41188,]


write.csv(file="pre_processed_training_data_11_12_6.csv", x=final_training_data,row.names = FALSE) 
write.csv(file="training_labels_11_12_6.csv", x=balanced_training_labels,row.names = FALSE)
write.csv(file="pre_processed_testing_data_11_12_6.csv", x=final_testing_data,row.names = FALSE) 

write.csv(file="pre_processed_validation_data_11_12_6.csv", x=scaled_validation,row.names = FALSE) 
write.csv(file="validation_labels_11_12_6.csv", x=validation_labels,row.names = FALSE)

print("Preprocessed files are generated....!")