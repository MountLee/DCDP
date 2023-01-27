setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("cpd_utils.R")

# install.packages("reticulate")
library("reticulate")
# for potential errors, see https://github.com/conda/conda/issues/11795
np <- import("numpy")

# install.packages("InspectChangepoint")
library(changepoints)
library(gglasso)
#setwd("/Users/darenw/Dropbox/vpcusum/code/scene1/")

library("MASS")



#### load data ####

K = 3
TT = K + 1
Delta = 50
n = Delta * TT
p = 20

delta = 5
theta = matrix(0, nrow = TT, ncol = p)
for (k in 1:TT){
  theta[k, (1 + 5 * (k - 1)):(5 * k)] = delta
}
kappa = sum((theta[1,] - theta[2,])**2)**0.5

data = np$load(paste("data_n", n, "_p", p, "_Delta", Delta, "_K", K, "_kappa", floor(kappa * 100), '.npz', sep = ''))

data$files
Y_train_list = data$f[['Y_train_list']]
X_train_list = data$f[['X_train_list']]
Y_test_list = data$f[['Y_test_list']]
X_test_list = data$f[['X_test_list']]
cp_truth_list = data$f[['cp_truth_list']]
beta = data$f[['beta']]



#### DP ####

#### check b=1
set.seed(0)

b = 1
Y_train = Y_train_list[b, , ]
X_train = (X_train_list[b, , ])
Y_test = Y_test_list[b, , ]
X_test = (X_test_list[b, , ])

X_all = array(0, dim = c(2 * n, p))
X_all[seq(1, 2 * n, 2), ] = X_train
X_all[seq(2, 2 * n, 2), ] = X_test

Y_all = rep(0, n)
Y_all[seq(1, 2 * n, 2)] = Y_train
Y_all[seq(2, 2 * n, 2)] = Y_test




gamma_set = c(0.01, 0.1)
lambda_set = c(0.01, 0.1)
temp = CV.search.DP.regression(y = Y_all, X = X_all, gamma_set, lambda_set, delta = 2)
temp$test_error # test error result
# find the indices of gamma_set and lambda_set which minimizes the test error
min_idx = as.vector(arrayInd(which.min(temp$test_error), dim(temp$test_error)))

gamma_set[min_idx[1]]
lambda_set[min_idx[2]]
cpt_init = unlist(temp$cpt_hat[min_idx[1], min_idx[2]])

zeta_set = c(0.1, 1)
temp_zeta = CV.search.DP.LR.regression(Y_all, X_all, gamma_set[min_idx[1]],
                                       lambda_set[min_idx[2]], zeta_set, delta = 2, eps = 0.001)
min_zeta_idx = which.min(unlist(temp_zeta$test_error))
cpt_LR = local.refine.regression(cpt_init, Y_all, X_all, zeta = zeta_set[min_zeta_idx])


cp_estimate = floor(cpt_LR / 2)
cp_distance(cp_estimate, cp_truth_list[b, ])






#### B=100
set.seed(0)

B = 100

cp_estimate_list = c()
loc_error_list = rep(Inf, B)
K_list = rep(0, B)
run_time_list = rep(NA, B)


for (b in 1:B){
  Y_train = Y_train_list[b, , ]
  X_train = (X_train_list[b, , ])
  Y_test = Y_test_list[b, , ]
  X_test = (X_test_list[b, , ])
  
  #### the x in changepoints should be n-by-p
  
  X_all = array(0, dim = c(2 * n, p))
  X_all[seq(1, 2 * n, 2), ] = X_train
  X_all[seq(2, 2 * n, 2), ] = X_test
  
  Y_all = rep(0, n)
  Y_all[seq(1, 2 * n, 2)] = Y_train
  Y_all[seq(2, 2 * n, 2)] = Y_test
  
  gamma_set = c(0.01, 0.1)
  lambda_set = c(0.01, 0.1)
  
  start_time <- Sys.time()
  
  
  # sink("dplr_log.txt")
  temp = CV.search.DP.regression(y = Y_all, X = X_all, gamma_set, lambda_set, delta = 2)
  temp$test_error # test error result
  # find the indices of gamma_set and lambda_set which minimizes the test error
  min_idx = as.vector(arrayInd(which.min(temp$test_error), dim(temp$test_error)))
  
  gamma_set[min_idx[1]]
  lambda_set[min_idx[2]]
  cpt_init = unlist(temp$cpt_hat[min_idx[1], min_idx[2]])
  zeta_set = c(0.1, 1)
  temp_zeta = CV.search.DP.LR.regression(Y_all, X_all, gamma_set[min_idx[1]],
                                         lambda_set[min_idx[2]], zeta_set, delta = 2, eps = 0.001)
  min_zeta_idx = which.min(unlist(temp_zeta$test_error))
  cpt_LR = local.refine.regression(cpt_init, Y_all, X_all, zeta = zeta_set[min_zeta_idx])
  
  cp_estimate = floor(cpt_LR / 2)
  
  # sink()
  run_time = difftime(Sys.time(), start_time, units = "secs")
  
  cp_estimate_list[[b]] = cp_estimate
  loc_error_list[b] = cp_distance(cp_estimate, cp_truth_list[b, ])
  K_list[b] = length(cp_estimate)
  run_time_list[b] = run_time
  
  print(b)
}


{
  print(paste("n", n, ", p", p, ", Delta", Delta, ", K", K, ", kappa", floor(kappa * 100)))
  print(paste("loc error: ", mean(loc_error_list), ", std: ", var(loc_error_list)^0.5))
  print(paste("run time: ", mean(run_time_list), ", std: ", var(run_time_list)^0.5))
  print(paste(" < K*: ", sum(K_list < K), " = K*: ", sum(K_list == K), " > K*: ", sum(K_list > K)))
}





#### more cases ####

dplr_one_case <- function(Delta, K, p, delta_signal, delta_dp = 2, B_MC = 100, gamma_set = c(0.01, 0.1),
                          lambda_set = c(0.01, 0.1), zeta_set = c(0.1, 1)){
  # K = 3
  TT = K + 1
  # Delta = 50
  n = Delta * TT
  # p = 20
  
  # delta = 5
  theta = matrix(0, nrow = TT, ncol = p)
  for (k in 1:TT){
    theta[k, (1 + 5 * (k - 1)):(5 * k)] = delta_signal
  }
  kappa = sum((theta[1,] - theta[2,])**2)**0.5
  
  data = np$load(paste("data_n", n, "_p", p, "_Delta", Delta, "_K", K, "_kappa", floor(kappa * 100), '.npz', sep = ''))

  Y_train_list = data$f[['Y_train_list']]
  X_train_list = data$f[['X_train_list']]
  Y_test_list = data$f[['Y_test_list']]
  X_test_list = data$f[['X_test_list']]
  cp_truth_list = data$f[['cp_truth_list']]
  beta = data$f[['beta']]

  #### projected WBS ####
  set.seed(0)
  
  cp_estimate_list = c()
  loc_error_list = rep(Inf, B_MC)
  K_list = rep(0, B_MC)
  run_time_list = rep(NA, B_MC)
  
  
  for (b in 1:B_MC){
    Y_train = Y_train_list[b, , ]
    X_train = t(X_train_list[b, , ])
    Y_test = Y_test_list[b, , ]
    X_test = t(X_test_list[b, , ])
    
    #### the x in inspect should be p-by-n
    
    X_all = array(0, dim = c(p, 2 * n))
    X_all[, seq(1, 2 * n, 2)] = X_train
    X_all[, seq(2, 2 * n, 2)] = X_test
    
    Y_all = rep(0, n)
    Y_all[seq(1, 2 * n, 2)] = Y_train
    Y_all[seq(2, 2 * n, 2)] = Y_test
    
    # gamma_set = c(0.01, 0.1)
    # lambda_set = c(0.01, 0.1)
    
    start_time <- Sys.time()
    
    
    # sink("dplr_log.txt")
    temp = CV.search.DP.regression(y = Y_all, X = X_all, gamma_set, lambda_set, delta = delta_dp)
    temp$test_error # test error result
    # find the indices of gamma_set and lambda_set which minimizes the test error
    min_idx = as.vector(arrayInd(which.min(temp$test_error), dim(temp$test_error)))
    
    gamma_set[min_idx[1]]
    lambda_set[min_idx[2]]
    cpt_init = unlist(temp$cpt_hat[min_idx[1], min_idx[2]])
    # zeta_set = c(0.1, 1)
    temp_zeta = CV.search.DP.LR.regression(Y_all, X_all, gamma_set[min_idx[1]],
                                           lambda_set[min_idx[2]], zeta_set, delta = delta_dp, eps = 0.001)
    min_zeta_idx = which.min(unlist(temp_zeta$test_error))
    cpt_LR = local.refine.regression(cpt_init, Y_all, X_all, zeta = zeta_set[min_zeta_idx])
    
    cp_estimate = floor(cpt_LR / 2)
    
    # sink()
    run_time = difftime(Sys.time(), start_time, units = "secs")
    
    cp_estimate_list[[b]] = cp_estimate
    loc_error_list[b] = cp_distance(cp_estimate, cp_truth_list[b, ])
    K_list[b] = length(cp_estimate)
    run_time_list[b] = run_time
    
    print(b)
  }
  
  
  {
    print(paste("n", n, ", p", p, ", Delta", Delta, ", K", K, ", kappa", floor(kappa * 100)))
    print(paste("loc error: ", mean(loc_error_list), ", std: ", var(loc_error_list)^0.5))
    print(paste("run time: ", mean(run_time_list), ", std: ", var(run_time_list)^0.5))
    print(paste(" < K*: ", sum(K_list < K), " = K*: ", sum(K_list == K), " > K*: ", sum(K_list > K)))
  }
  
  return(list(cp_estimate_list = cp_estimate_list, loc_error_list = loc_error_list,
              K_list = K_list, run_time_list = run_time_list))
  
}


#### p = 20

res = dplr_one_case(Delta = 50, K = 3, p = 20, delta_signal = 5, delta_dp = 2, B_MC = 100, gamma_set = c(0.01, 0.1),
              lambda_set = c(0.01, 0.1), zeta_set = c(0.1, 1))
# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 1581"
# [1] "loc error:  0.04 , std:  0.196946385566932"
# [1] "run time:  16.9785236406326 , std:  0.518191369453406"
# [1] " < K*:  0  = K*:  100  > K*:  0"



res = dplr_one_case(Delta = 50, K = 3, p = 20, delta_signal = 1, delta_dp = 2, B_MC = 100, gamma_set = c(0.01, 0.1),
              lambda_set = c(0.01, 0.1), zeta_set = c(0.1, 1))

# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  0.05 , std:  0.21904291355759"
# [1] "run time:  12.8094676399231 , std:  0.471755879762532"
# [1] " < K*:  0  = K*:  100  > K*:  0"

#### p = 100

res = dplr_one_case(Delta = 50, K = 3, p = 100, delta_signal = 5, delta_dp = 2, B_MC = 100, gamma_set = c(0.01, 0.1),
              lambda_set = c(0.01, 0.1), zeta_set = c(0.1, 1))

# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 1581"
# [1] "loc error:  0.01 , std:  0.1"
# [1] "run time:  220.313827433586 , std:  16.8009739192564"
# [1] " < K*:  0  = K*:  98  > K*:  2"

res = dplr_one_case(Delta = 50, K = 3, p = 100, delta_signal = 1, delta_dp = 2, B_MC = 100, gamma_set = c(0.01, 0.1),
              lambda_set = c(0.01, 0.1), zeta_set = c(0.1, 1))

# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  0.22 , std:  2.00292715087623"
# [1] "run time:  84.3718576788902 , std:  5.72146110208264"
# [1] " < K*:  0  = K*:  99  > K*:  1"

