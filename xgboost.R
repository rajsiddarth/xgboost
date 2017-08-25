#Implementation of xgboost in R
rm(list=ls(all=TRUE))
library(RCurl)

data=read.table(text = getURL("https://raw.githubusercontent.com/rajsiddarth/xgboost_gradientboost/master/Dataset.csv"), header=T, sep=',',
                col.names = c('ID', 'age', 'exp', 'inc', 
                              'zip', 'family', 'ccavg', 'edu', 
                              'mortgage', 'loan', 'securities', 
                              'cd', 'online', 'cc'))
#Removing the id, zip and experience

data=subset(data,select = -c(ID,zip,exp))

#Numeric attributes : age,inc,family,CCAvg,Mortgage
#Categorical: Education,Securities account,CD Account,Online,Credit card
#Target Variable: Personal Loan
num_data=data.frame(sapply(data[c('age','inc','family','ccavg')],function(x){as.numeric(x)}))
categ_attributes=c('edu','securities','cd','online')
categ_data=data.frame(sapply(data[categ_attributes],function(x){as.factor(x)}))
loan=as.factor(data$loan)
data=cbind(num_data,categ_data,loan)
str(data)

# Standardizing the numeric data
library(vegan)
final_Data1 = decostand(data[,names(num_data)], "range") 

# Convert all categorical attributes to numeric using dummies
library(dummies)
final_Data2=dummy.data.frame(data[categ_attributes])
final_data=cbind(final_Data1,final_Data2,loan)
str(final_data)

# Build the xgboost classification model.

ind_Attr = setdiff(names(final_data), "loan")

# Divide the data into test, train and eval
set.seed(123)
rowIDs = 1:nrow(final_data)
train_RowIDs =  sample(rowIDs, length(rowIDs)*0.6)
test_RowIDs = sample(setdiff(rowIDs, train_RowIDs), length(rowIDs)*0.2)
eval_RowIDs = setdiff(rowIDs, c(train_RowIDs, test_RowIDs))

train_Data = final_data[train_RowIDs,]
test_Data = final_data[test_RowIDs,]
eval_Data = final_data[eval_RowIDs,]

#install.packages("xgboost")
library(xgboost)

dtrain = xgb.DMatrix(data = as.matrix(train_Data[,ind_Attr]),
                     label = train_Data$loan)
model = xgboost(data = dtrain, max.depth = 2, 
                eta = 1, nthread = 2, nround = 2, 
                objective = "binary:logistic", verbose = 1)

# objective = "binary:logistic": we will train a binary classification model ;
# max.deph = 2: the trees won't be deep, because our case is very simple ;
# nthread = 2: the number of cpu threads we are going to use;
# nround = 2: there will be two passes on the data
# eta = 1: It controls the learning rate
# verbose = 1: print evaluation metric

# Both xgboost (simple) and xgb.train (advanced) functions train models.

# Because of the way boosting works, there is a time when having too many rounds lead to an overfitting. One way to measure progress in learning of a model is to provide to XGBoost a second dataset already classified. Therefore it can learn on the first dataset and test its model on the second one. Some metrics are measured after each round during the learning.

#Use watchlist parameter. It is a list of xgb.DMatrix, each of them tagged with a name.
dtest = xgb.DMatrix(data = as.matrix(test_Data[,ind_Attr]),
                    label = test_Data$loan)

watchlist = list(train=dtrain, test=dtest)

model = xgb.train(data=dtrain, max.depth=2,
                  eta=1, nthread = 2, nround=5, 
                  watchlist=watchlist,
                  eval.metric = "error", 
                  objective = "binary:logistic")
# eval.metric allows us to monitor two new metrics for each round, logloss and error.

importance <- xgb.importance(feature_names = ind_Attr, model = model)
print(importance)
xgb.plot.importance(importance_matrix = importance)

# Gain is the improvement in accuracy brought by a feature to the branches it is on. 
# Cover measures the relative quantity of observations concerned by a feature.
# Frequency is the number of times a feature is used in all generated trees. 

# save model to binary local file
xgb.save(model, "xgboost.model")
rm(model)

# load binary model to R
model <- xgb.load("xgboost.model")

# predict
pred <- predict(model, as.matrix(eval_Data[,ind_Attr]))

# size of the prediction vector
print(length(pred))

# limit display of predictions to the first 10
print(head(pred))

# The numbers we get are probabilities that a datum will be classified as 1. 
# Therefore, we will set the rule that if this probability for a 
# specific datum is > 0.5 then the observation is classified as 1 (or 0 otherwise).

prediction <- as.numeric(pred > 0.5)
print(head(prediction))