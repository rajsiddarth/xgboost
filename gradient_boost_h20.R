#Implementation of gradient boosting in h20 

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

#Final data
data=cbind(num_data,categ_data,loan)
str(data)

#Dividing into train and test
library(caTools)
index=sample.split(data$loan,SplitRatio = 0.7)
train=data[index,]
test=data[!index,]
ind_variables=setdiff(names(data),"loan")

# Load H2o library
library(h2o)

# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init(nthreads = -1, max_mem_size = "2g")

# Import a local R train data frame to the H2O cloud
train_data= as.h2o(x = train, destination_frame = "train_data")

# Prepare the parameters for the for H2O gbm grid search
ntrees= c(5, 10, 15, 20, 25, 30)
maxdepth=c(2, 3, 4)
learnrate=c(0.01, 0.05, 0.1, 0.15 ,0.2, 0.25)
hyper_parameters=list(ntrees = ntrees,max_depth = maxdepth,learn_rate = learnrate)

# Build H2O GBM with grid search
grid_GBM= h2o.grid(algorithm = "gbm", grid_id = "grid_GBM_data",hyper_params = hyper_parameters, 
                     y = "loan", x = setdiff(names(train_data), "loan"),
                     training_frame = train_data)

summary(grid_GBM)

# Fetch GBM grid models
grid_GBM_models=lapply(grid_GBM@model_ids, 
                          function(model_id) { h2o.getModel(model_id) })

# Function to find the best model with respective to AUC
find_Best_Model=function(grid_models){
  best_model = grid_models[[1]]
  best_model_AUC = h2o.auc(best_model)
  for (i in 2:length(grid_models)) 
  {
    temp_model = grid_models[[i]]
    temp_model_AUC = h2o.auc(temp_model)
    if(best_model_AUC < temp_model_AUC)
    {
      best_model = temp_model
      best_model_AUC = temp_model_AUC
    }
  }
  return(best_model)
}

# Find the best model by calling find_Best_Model Function
best_GBM_model = find_Best_Model(grid_GBM_models)

# Get the auc of the best GBM model
best_GBM_model_AUC = h2o.auc(best_GBM_model)

# Examine the performance of the best model
best_GBM_model

# View the specified parameters of the best model
best_GBM_model@parameters

# Important Variables.
varImp_GBM = h2o.varimp(best_GBM_model)

# Import a local R test data frame to the H2O cloud
test_data=as.h2o(x = test, destination_frame = "test_data")

# Predict on same training data set
predict= h2o.predict(best_GBM_model, 
                          newdata = test_data[,setdiff(names(test_data), "loan")])
       
data_GBM = h2o.cbind(test_data[,"loan"], predict)
                    
# Copy predictions from H2O to R
pred_GBM = as.data.frame(data_GBM)

# Shutdown H2O
h2o.shutdown(F)

# Hit Rate and Penetration calculation
conf_Matrix_GBM = table(pred_GBM$loan, pred_GBM$predict) 

Accuracy = (conf_Matrix_GBM[1,1]+conf_Matrix_GBM[2,2])/sum(conf_Matrix_GBM)
cat("accuracy on test data= ",round(Accuracy,3)*100)
