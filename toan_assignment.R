#remove environment
rm(list=ls())
# Load packages
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm

#you need to change to your files locations
train <- read.csv("~/Documents/R/data/train.csv")
test <- read.csv("~/Documents/R/data/test.csv")
train$y <- ifelse(train$y == "yes", 1, 0)
test$y <- ifelse(test$y == "yes", 1, 0)
sum(is.na(train))
summary(train)


# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)
rf_model <-randomForest(factor(y) ~ age + job  + education 
                        + month + day_of_week + duration + campaign + pdays + poutcome + emp.var.rate 
                        + cons.price.idx + cons.conf.idx + euribor3m + nr.employed,
                        data = train)

#show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

# Predict using the test set
prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(id = test$id, prediction = prediction)

# Write the solution to file
write.csv(solution, file = '~/Documents/R/data/rf_mod_Solution.csv', row.names = F)
