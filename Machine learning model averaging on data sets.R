#Load packages
library(Rsolnp)
library(matrixStats)
library(caret)
library(caretEnsemble)
library(ModelMetrics)
library(glmnet)
library(ranger)
library(e1071)
library(ada)
library(ranger)
library(foreach)
library(doParallel)
library(doRNG)
library(tidyverse)
library(robustHD)

#First load a data set from "data sets for machine learning"

#Create cluster for parallel computation
parallel::detectCores()
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = my.cluster)

#Load required packages inside each parallel process
clusterEvalQ(my.cluster, {
  library(Rsolnp)
  library(matrixStats)
  library(caret)
  library(caretEnsemble)
  library(ModelMetrics)
  library(glmnet)
  library(ranger)
  library(e1071)
  library(ada)
  library(ranger)
})



#Start simulation
n_trials <- 100
set.seed(12345) #Set seed for parallel computations
run <- foreach(trials = 1:n_trials) %dorng% {
  
  #Divide data into training and test set
  n <- nrow(data)
  
  data <- data[sample(nrow(data)), ]
  data_train <- data[1:round(n*0.85),]
  data_test <- data[(round(n*0.85) + 1):n,]
  
  #Caret train control
  my_control <- trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 1,
    savePredictions="final",
    classProbs=TRUE,
    index=createMultiFolds(data_train$target, k = 5, times = 1),
    summaryFunction = log.loss.capped
  )
  
  #Defines the models
  num_of_models <- 5
  model_list <- caretList(
    target~., data=data_train,
    trControl=my_control,
    methodList=c("svmRadial", "glm", "knn", "glmnet", "rf"),
    maximize = FALSE
  )

  model_optimism <- c(length = num_of_models)
  
  fitted_predictions <- matrix(nrow = nrow(data_train), ncol = num_of_models)
  
  cv_log_scores <- c(length = num_of_models)
  
  for(i in 1:num_of_models){
    preds <- predict(model_list[[i]], newdata = data_train, type = "prob")
    outcomes <- cbind(1:nrow(data_train), as.numeric(data_train[,1]))
    fitted_log_scores <- -sum(log(preds[outcomes]))
    cv_log_scores[i] <- min(model_list[[i]]$results$logLoss)*5
    fitted_predictions[,i] <- preds[outcomes]
    model_optimism[i] <- cv_log_scores[i] - fitted_log_scores
  }
  
  #Prior model weights. Note that the prior is just the unnormalized optimism estimates. 
  #To avoid numerical instability, we don't convert these estimates to probabilities.
  prior <- model_optimism - min(model_optimism)
  prior <- prior
  
  model_weights <- divergence_weights(pointwise = fitted_predictions, prior = prior)
  
  #Matrix for saving the leave-out predictions from all the models
  ldp_pointwise <- matrix(nrow = nrow(data_train), ncol = num_of_models)
  
  for(j in 1:nrow(data_train)){
    for(i in 1:num_of_models){
      outcome <- model_list[[i]]$pred[j,]$obs
      outcome <- as.character(outcome)
      index <- which.max(colnames(model_list[[i]]$pred[j,]) == outcome)
      pred <- as.numeric(model_list[[i]]$pred[j,][index])
      ldp_pointwise[j, i] <- pred
      }
    }
  
  #Stacking with the log score model weights 
  ldp_model_weights <- stacking_weights(pointwise = ldp_pointwise)
  
  #Calculate negative exponentiated model weights
  cv_weights <- cv_log_scores - min(cv_log_scores)
  cv_weights <- exp(-cv_weights)
  cv_weights <- cv_weights/sum(cv_weights)

  
  #General linear model meta-learner
  ensemble <- caretStack(model_list,
                         method="glm",
                         trControl= my_control,
                         metric = "logLoss",
                         maximize = FALSE)
  
  #Elastic net meta-learner
  ensemble2 <- caretStack(model_list, 
                          method="glmnet",  
                          trControl= my_control, 
                          metric = "logLoss", 
                          maximize = FALSE)

  #Gradient boosting machine meta-learner
  ensemble3 <- caretStack(model_list,
                          method="gbm",
                          trControl= my_control,
                          metric = "logLoss",
                          maximize = FALSE)

  #Save the predictions made by the ensembles
  ensemb_prob <-  abs(as.numeric(data_test[,1]) - 1 - predict(ensemble, newdata = data_test, type = "prob"))
  ensemb2_prob <-  abs(as.numeric(data_test[,1]) - 1 - predict(ensemble2, newdata = data_test, type = "prob"))
  ensemb3_prob <-  abs(as.numeric(data_test[,1]) - 1 - predict(ensemble3, newdata = data_test, type = "prob"))

  #Predictions made by the models on the test set
  test_predictions <- matrix(nrow = nrow(data_test), ncol = num_of_models)
  for(i in 1:num_of_models){
    preds <- predict(model_list[[i]], newdata = data_test, type = "prob")
    outcomes <- cbind(1:nrow(data_test), as.numeric(data_test[,1]))
    test_predictions[,i] <- preds[outcomes]
  }
  
  #Calculate the (capped) log score of each model weighting method on the test set
  div_log_scores <-   mean(-capped.log(test_predictions%*%model_weights))
  stack_log_scores <-  mean(-capped.log(test_predictions%*%ldp_model_weights))
  cv_log_scores <-   mean(-capped.log(test_predictions%*%cv_weights))
  ensemble_log_scores <-   mean(-capped.log(ensemb_prob))
  ensemble2_log_scores <-   mean(-capped.log(ensemb2_prob))
  ensemble3_log_scores <-   mean(-capped.log(ensemb3_prob))
  
  
  #Collect all metrics in a table
  result <- cbind(div_log_scores,   ensemble_log_scores,  stack_log_scores, ensemble2_log_scores, ensemble3_log_scores, cv_log_scores)

  return(result)
}
parallel::stopCluster(cl = my.cluster) #End trials

#Average results from all trials
sum <- 0
for(i in 1:n_trials){
  sum <- sum + run[[i]]
}
average <- sum/n_trials
average