#This is the code used to conduct the simulation in Section 4.1: 
#Simulation experiment
#Please source the functions in "Optimization functions" and "Convenience functions"

#Load packages
library("foreach")
library("doParallel")
library("doRNG")
library("tidyverse")
library("robustHD")
library("Rsolnp")

#Create cluster for parallel computation
n.cores <- parallel::detectCores() - 1
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = my.cluster)

#Load required packages inside each parallel process
clusterEvalQ(my.cluster, {
  library(matrixStats)
  library(Rsolnp)
})


#Start simulation
n_trials <- 1000

set.seed(123) #Set seed for parallel computations
run <- foreach(trials = 1:n_trials) %dorng% {
  sample_size <- c(10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200)
  test_run <- matrix(nrow = 13, ncol = 4)
  
  for(number in 1:13){
    
    #Randomly generate coefficients for data-generating distribution
    num_of_var <- 20
    intercept <- rnorm(1, 0, 2)
    a <- rnorm(num_of_var, 0, 0.5)
    
    #Randomly generate data
    X <- cbind(sapply(rep(0, num_of_var), rnorm, n = 1000, sd = 1))
    error <- rnorm(1000, 0, 5)
    Y <- X%*%a + intercept + error
    d <- cbind(Y, X)
    d <- as.data.frame(d)
    names(d)[1] <- "target"
    n <- nrow(d)
    i <- sample_size[number]
    
    #Extract predictors from dataset
    predictors <- predictor_extractor(d)
    
    #Randomly generate models
    model_list <- model_space_creator(predictors, 15)

    #Divide data into training set with i observations and test set with 200 observations 
    d_training <- d[1:i,]
    d_testing <- d[(i+1):(i + 200),]
    
    #Fit models on training set
    model_fit <- lm_fit(model_list, d_training)
    
    #Create AIC-penalized prior
    prior <- aic_prior(model_fit, i)
    
    #Create pointwise matrix of maximum likelihood scores
    pointwise <- pointwise_matrix(model_fit, d_training)
    
    #Calculate divergence-based model weights
    model_weights1 <- divergence_weights(pointwise = pointwise, prior = prior)
    
    #Create leave-10-out matrix of model predictors
    lo_matrix <- leave_k_out_density_matrix(model_list, data = d_training, nfold = 10)
    
    #Calculate stacking weights
    model_weights2 <- stacking_weights(pointwise = exp(lo_matrix))
    #model_weights2 <- loo::stacking_weights(lpd_point = lo_matrix)
    
    #Calculate Akaike weights
    aic_scores <- aic(model_fit, i)
    aic_scores <- (aic_scores - min(aic_scores))*0.5
    aic_scores <- exp(-aic_scores)
    model_weights4 <- aic_scores/sum(aic_scores)
    
    #Calculate predictions using each method on test set
    divergence_pred <- fitmodel_predictions(fitmodel_list = model_fit, model_probabilities = model_weights1, test_data = d_testing)
    stacking_pred <- fitmodel_predictions(model_fit, model_probabilities = model_weights2, d_testing)
    aic_pred <- fitmodel_predictions(model_fit, model_probabilities = model_weights3, d_testing)

    #Calculate RMSE on test set
    result <- c(sqrt(mean((divergence_pred - d_testing$target)^2)), 
                sqrt(mean((stacking_pred - d_testing$target)^2)),
                sqrt(mean((aic_pred - d_testing$target)^2)))
    
    test_run[number,] <- result
  }
  return(test_run)
}
parallel::stopCluster(cl = my.cluster) #End simulation

#Calculate averages
sum <- 0
for(i in 1:n_trials){
  sum <- sum + run[[i]]
}
average <- sum/n_trials

#Plot results
library("tidyverse")
x <- c(10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 150, 200)
results <- data.frame(data = x, divergence_weights = as.vector(average[,1]), stacking_weights = as.vector(average[, 2]), aic_weights = as.vector(average[, 3]))
results.tidy <- gather(results, Averaging_method, Log_score, -data)

ggplot(results.tidy, aes(x = data, y = Log_score, col = Averaging_method)) + 
  geom_point(shape = 1, alpha = 0.5) + 
  coord_cartesian(
    ylim = c(5.3, 6.7),
    expand = TRUE,
    default = FALSE,
    clip = "on" 
  ) +
  theme_bw() +
  theme(text = element_text(family="serif")) +
  theme(text = element_text(size=15)) +
  theme(legend.position="none", legend.title = element_blank()) +
  xlab("Sample size") +
  ylab("RMSE") +
  coord_fixed(ratio = 125) +
  geom_line(aes(col = Averaging_method)) +
  scale_color_manual(labels = c("AIC weighting", "Divergence-based weighting", "Stacking"), values = c("green", "blue", "red")) 
