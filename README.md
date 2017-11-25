### Bank Marketing: predict if the client will subscribe a term deposit###

In this project, we implement a set of machine learning method for predicting if the client will subscribe a term deposit. The data set can be found [here](https://data.world/uci/bank-marketing). The whole data set are devided into two part training set and testing set, which could be found in *data* directory.

#### Data processing
- Handling missing values(found in random-forest mode)
- Feature engineering (found in random-forest mode)
- Unbalanced dataset Solving (found in DeepLearning mode, the data preprocessing part)
- Standardization data (found in DeepLearning mode)
- Dataset is a combination of continuous and categorical variables (found in DeepLearning mode)

#### Machine Learning Models
We have implemented several models, including:
a. Decision Tree
b. Naive Bayes
c. SVM
d. Neural Net
e. Random Forest
f. Conditional Random Forests (cforest)

#### Models Evaluations
The evaluation of the performance of those models are based on Accurancy and Matthews correlation coefficient (MCC).
Overall, Cforest generate best results.

#### Enviroment requirement
Enviroment of cforest, random forest, Naive Bayes, SVM, Decision tree:
- Ubuntu 16.04 with 64 GB of RAM
- rstudio on Linux
- with all listed library

Requirements for Deep Learning Model
- Tensorflow 1..3 (GPU version is preferred)
- Numpy 1.13.1
- matplotlib 2.1.0
- sklearn 0.19.1
- R studio and listed libraries to run DL_Preprocessing.R file