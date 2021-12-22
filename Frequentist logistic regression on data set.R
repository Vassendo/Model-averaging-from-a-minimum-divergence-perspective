#This is the code used for the data analysis in Section 4.2: Predicting 
#survival among patients with heart failure
#Please source the functions in "Optimization functions" and "Convenience functions"

#Load packages
library("foreach")
library("doParallel")
library("doRNG")
library("tidyverse")
library("robustHD")
library("splines")


#Load data set, clean, and standardize
data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
data <- data[, c(13, 1:12)]
data <- data[, -13] #Remove follow-up time
data <- data[complete.cases(data),]
predictors <- standardize(data[,-1])
data <- cbind(data[,1], predictors)
names(data)[1] <- "target"
n <- nrow(data)

#Extract predictors from dataset
predictors <- predictor_extractor(d)

#Create splines 
B_ef_fraction <- bs(data$ejection_fraction, intercept = FALSE)
B_age <- bs(data$age, intercept = FALSE)
colnames(B_ef_fraction) <- c("EF1", "EF2", "EF3")
colnames(B_age) <- c("AGE1", "AGE2", "AGE3")
data <- cbind(data, B_ef_fraction, B_age)

#Create interactions
age_bp_int <- (data$age)*(data$high_blood_pressure)
age_ef_int <- (data$age)*(data$ejection_fraction)
age_smoking_int <- (data$age)*(data$smoking)
data <- cbind(data, age_bp_int, age_ef_int, age_smoking_int)

#Create model space
num_of_models <- 5

model_list <- list(length = num_of_models)

model_list[[1]] <- target ~ age + anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + platelets + serum_creatinine + serum_sodium + sex + smoking
model_list[[2]] <- target ~ age + high_blood_pressure + age_bp_int
model_list[[3]] <- target ~ AGE1 + AGE2 + AGE3 + high_blood_pressure + age_bp_int
model_list[[4]] <- target ~ age + ejection_fraction + age_ef_int
model_list[[5]] <- target ~ AGE1 + AGE2 + AGE3 + EF1 + EF2 + EF3 + age_ef_int

#Set up cluster for parallel computing
parallel::detectCores()
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = my.cluster)

#Load required packages inside each parallel process
clusterEvalQ(my.cluster, {
  library(matrixStats)
  library(ModelMetrics)
  library(Rsolnp)
  library(splines)
})

#Start trials
n_trials <- 1000

set.seed(123) #Set seed for parallel computations
run <- foreach(trials = 1:n_trials) %dorng% {
  #Randomly shuffle data
  d <- data[sample(nrow(data)), ]
  
  #Training data set size
  i <- round(n*0.85)
  
  #Divide data into training set with i observations and test set with remaining observations 
  d_training <- d[1:i,]
  d_testing <- d[(i+1):n,]
  
  #Fit models on training set
  model_fit <- glm_fit(model_list, d_training)
  
  #Calculate corrected AIC estimates of model optimism (which function as prior model weights)
  prior <- aic_prior(model_fit, i)
  
  #Create pointwise matrix of maximum likelihood scores
  pointwise <- glm_pointwise_matrix(model_fit, d_training)
  
  #Calculate divergence-based model weights
  model_weights1 <- divergence_weights(pointwise = pointwise, prior = prior)
  
  #Create leave-10-out matrix of model log-predictors
  lo_matrix <- glm_leave_k_out_matrix(model_list, data = d_training, nfold = 5)
  
  #Calculate stacking weights
  model_weights2 <- stacking_weights(pointwise = exp(lo_matrix))
  
  #Calculate negative exponentiated weights with 10-fold cross validation 
  lo_scores <- colSums(-lo_matrix) - min(colSums(-lo_matrix))
  lo_scores <- exp(-lo_scores)
  model_weights3 <- lo_scores/sum(lo_scores)
  
  #Calculate Akaike weights
  aic_scores <- aic(model_fit, i)
  aic_scores <- (aic_scores - min(aic_scores))*0.5
  aic_scores <- exp(-aic_scores)
  model_weights4 <- aic_scores/sum(aic_scores)
  
  #Calculate predictions using each method on test set
  divergence_pred <- fitmodel_predictions(fitmodel_list = model_fit, model_probabilities = model_weights1, test_data = d_testing)
  stacking_pred <- fitmodel_predictions(model_fit, model_probabilities = model_weights2, d_testing)
  aic_pred <- fitmodel_predictions(model_fit, model_probabilities = model_weights3, d_testing)

  #Calculate Brier scores on test set
  Brier <- c(mean((divergence_pred - d_testing$target)^2),
             (mean((stacking_pred - d_testing$target)^2)),
             (mean((aic_pred - d_testing$target)^2)))
  #Calculate Log scores on test set
  Log <- c(mean(-log(glm_probability(divergence_pred, d_testing))),
           mean(-log(glm_probability(stacking_pred, d_testing))),
           mean(-log(glm_probability(aic_pred, d_testing))))
  
  #Calculate Accuracy scores on test set
  score1 <- 100*sum(as.numeric(as.numeric(divergence_pred > 0.5) == d_testing[, 1]))/nrow(d_testing)
  score2 <- 100*sum(as.numeric(as.numeric(stacking_pred > 0.5) == d_testing[, 1]))/nrow(d_testing)
  score3 <- 100*sum(as.numeric(as.numeric(aic_pred > 0.5) == d_testing[, 1]))/nrow(d_testing)

  
  Accuracy <- c(score1, score2, score3)
  
  #Calculate AUC scores on test set
  score1 <- ModelMetrics::auc(d_testing[,1], divergence_pred)
  score2 <- ModelMetrics::auc(d_testing[,1], stacking_pred)
  score3 <- ModelMetrics::auc(d_testing[,1], aic_pred)


  AUC <- c(score1, score2, score3)
  
  #Collect all metrics in a table
  result <- cbind(Log, Brier, AUC, Accuracy)
  row.names(result) <- c("Divergence_based weighting", "Stacking", "AIC weighting")
  
  return(result)
  
}
parallel::stopCluster(cl = my.cluster) #End trials

#Calculate averages
sum <- 0
for(i in 1:n_trials){
  sum <- sum + run[[i]]
}
average <- sum/n_trials
average