rm(list=ls(all=TRUE))

setwd("C:/Users/jeevan/Desktop/Lab")

attr = c('id', 'age', 'exp', 'inc', 'zip', 'family', 
         'ccavg', 'edu', 'mortgage', 'loan', 
         'securities', 'cd', 'online', 'cc')

# Read the data using csv file
data = read.csv(file = "UniversalBank.csv", 
                header = TRUE, col.names = attr)

# Removing the id, zip and experience. 
drop_Attr = c("id", "zip", "exp")
attr = setdiff(attr, drop_Attr)
data = data[, attr]
rm(drop_Attr)

# Convert attribute to appropriate type  
cat_Attr = c("family", "edu", "securities", 
             "cd", "online", "cc", "loan")
num_Attr = setdiff(attr, cat_Attr)
rm(attr)

cat_Data <- data.frame(sapply(data[,cat_Attr], as.factor))
num_Data <- data.frame(sapply(data[,num_Attr], as.numeric))

data = cbind(num_Data, cat_Data)
rm(cat_Data, num_Data, cat_Attr, num_Attr)

# Do the summary statistics and check for missing values and outliers.
summary(data)

# Divide the data in to test and train
set.seed(123)
train_rowIDs = sample(1:nrow(data), nrow(data)*.8)
train = data[train_rowIDs,] 
test = data[-train_rowIDs,] 

rm(data, train_rowIDs)

# Load H2o library
library(h2o)

# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init(nthreads = -1, max_mem_size = "2g")

# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = train, destination_frame = "train.hex")


# Prepare the parameters for the for H2O gbm grid search
ntrees_opt <- c(5, 10, 15, 20, 30)
maxdepth_opt <- c(2, 3, 4, 5)
learnrate_opt <- c(0.01, 0.05, 0.1, 0.15 ,0.2, 0.25)
hyper_parameters <- list(ntrees = ntrees_opt, 
                         max_depth = maxdepth_opt, 
                         learn_rate = learnrate_opt)

# Build H2O GBM with grid search
grid_GBM <- h2o.grid(algorithm = "gbm", grid_id = "grid_GBM.hex",
                     hyper_params = hyper_parameters, 
                     y = "loan", x = setdiff(names(train.hex), "loan"),
                     training_frame = train.hex)

# Remove unused R objects
rm(ntrees_opt, maxdepth_opt, learnrate_opt, hyper_parameters)

# Get grid summary
summary(grid_GBM)

# Fetch GBM grid models
grid_GBM_models <- lapply(grid_GBM@model_ids, 
                          function(model_id) { h2o.getModel(model_id) })

# Function to find the best model with respective to AUC
find_Best_Model <- function(grid_models){
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

rm(grid_GBM_models)

# Get the auc of the best GBM model
best_GBM_model_AUC = h2o.auc(best_GBM_model)

# Examine the performance of the best model
best_GBM_model

# View the specified parameters of the best model
best_GBM_model@parameters

# Important Variables.
varImp_GBM <- h2o.varimp(best_GBM_model)

# Import a local R test data frame to the H2O cloud
test.hex <- as.h2o(x = test, destination_frame = "test.hex")


# Predict on same training data set
predict.hex = h2o.predict(best_GBM_model, 
                          newdata = test.hex[,setdiff(names(test.hex), "loan")])
       
data_GBM = h2o.cbind(test.hex[,"loan"], predict.hex)
                    
# Copy predictions from H2O to R
pred_GBM = as.data.frame(data_GBM)

# Shutdown H2O
h2o.shutdown(F)

# Hit Rate and Penetration calculation
conf_Matrix_GBM = table(pred_GBM$loan, pred_GBM$predict) 

Accuracy = (conf_Matrix_GBM[1,1]+conf_Matrix_GBM[2,2])/sum(conf_Matrix_GBM)
