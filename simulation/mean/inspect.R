setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# install.packages("reticulate")
library("reticulate")
# for potential errors, see https://github.com/conda/conda/issues/11795
np <- import("numpy")

# install.packages("InspectChangepoint")
library(InspectChangepoint)
source("cpd_utils.R")

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
Y_test_list = data$f[['Y_test_list']]
cp_truth_list = data$f[['cp_truth_list']]
theta = data$f[['theta']]



#### projected WBS ####


set.seed(0)
# determine the threshold
threshold <- compute.threshold(2 * n,p)

B = 100

cp_estimate_list = c()
loc_error_list = rep(Inf, B)
K_list = rep(0, B)
run_time_list = rep(NA, B)

for (b in 1:B){
  Y_train = Y_train_list[b, , ]
  Y_test = Y_test_list[b, , ]
  Y_all = array(0, dim = c(2 * n, p))
  Y_all[seq(1, 2 * n, 2), ] = Y_train
  Y_all[seq(2, 2 * n, 2), ] = Y_test
  #### the x in inspect should be p-by-n
  start.time <- Sys.time()
  res <- inspect(x = t(Y_all), threshold = threshold)  
  run_time <- Sys.time() - start.time
  
  cp_estimate = round(res$changepoints[, 1] / 2)
  cp_estimate_list[[b]] = cp_estimate
  loc_error_list[b] = cp_distance(cp_estimate, cp_truth_list[b, ])
  K_list[b] = length(cp_estimate)
  run_time_list[b] = run_time
}


{
  print(paste("n", n, ", p", p, ", Delta", Delta, ", K", K, ", kappa", floor(kappa * 100)))
  print(paste("loc error: ", mean(loc_error_list), ", std: ", var(loc_error_list)^0.5))
  print(paste("run time: ", mean(run_time_list), ", std: ", var(run_time_list)^0.5))
  print(paste(" < K*: ", sum(K_list < K), " = K*: ", sum(K_list == K), " > K*: ", sum(K_list > K)))
}


# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 1581"
# [1] "loc error:  0.54 , std:  4.46404264850559"
# [1] "run time:  0.00754257917404175 , std:  0.00158527985443491"
# [1] " < K*:  0  = K*:  96  > K*:  4"


#### more cases ####

inspect_one_case <- function(Delta, K, p, delta){
  
  # K = 3
  TT = K + 1
  # Delta = 50
  n = Delta * TT
  # p = 20
  
  # delta = 1
  theta = matrix(0, nrow = TT, ncol = p)
  for (k in 1:TT){
    theta[k, (1 + 5 * (k - 1)):(5 * k)] = delta
  }
  kappa = sum((theta[1,] - theta[2,])**2)**0.5
  
  data = np$load(paste("data_n", n, "_p", p, "_Delta", Delta, "_K", K, "_kappa", floor(kappa * 100), '.npz', sep = ''))

  Y_train_list = data$f[['Y_train_list']]
  Y_test_list = data$f[['Y_test_list']]
  cp_truth_list = data$f[['cp_truth_list']]
  theta = data$f[['theta']]
  
  
  set.seed(0)
  # determine the threshold
  threshold <- compute.threshold(2*n,p)
  
  B = 100
  
  cp_estimate_list = c()
  loc_error_list = rep(Inf, B)
  K_list = rep(0, B)
  run_time_list = rep(NA, B)
  
  for (b in 1:B){
    Y_train = Y_train_list[b, , ]
    Y_test = Y_test_list[b, , ]
    Y_all = array(0, dim = c(2 * n, p))
    Y_all[seq(1, 2 * n, 2), ] = Y_train
    Y_all[seq(2, 2 * n, 2), ] = Y_test
    #### the x in inspect should be p-by-n
    start.time <- Sys.time()
    res <- inspect(x = t(Y_all), threshold = threshold)  
    run_time <- Sys.time() - start.time
    
    cp_estimate = round(res$changepoints[, 1] / 2)
    cp_estimate_list[[b]] = cp_estimate
    loc_error_list[b] = cp_distance(cp_estimate, cp_truth_list[b, ])
    K_list[b] = length(cp_estimate)
    run_time_list[b] = run_time
  }
  
  
  {
    print(paste("n", n, ", p", p, ", Delta", Delta, ", K", K, ", kappa", floor(kappa * 100)))
    print(paste("loc error: ", mean(loc_error_list), ", std: ", var(loc_error_list)^0.5))
    print(paste("run time: ", mean(run_time_list), ", std: ", var(run_time_list)^0.5))
    print(paste(" < K*: ", sum(K_list < K), " = K*: ", sum(K_list == K), " > K*: ", sum(K_list > K)))
  }
}


#### p = 20

inspect_one_case(Delta = 50, K = 3, p = 20, delta = 1)
# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  3.13 , std:  5.5023502783682"
# [1] "run time:  0.0083748197555542 , std:  0.00256042306825878"
# [1] " < K*:  0  = K*:  67  > K*:  33"

inspect_one_case(Delta = 50, K = 3, p = 20, delta = 0.5)
# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 158"
# [1] "loc error:  6.85 , std:  7.53359479299561"
# [1] "run time:  0.00878916501998901 , std:  0.00933443248023834"
# [1] " < K*:  0  = K*:  78  > K*:  22"


#### p = 100

inspect_one_case(Delta = 50, K = 3, p = 100, delta = 5)
# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 1581"
# [1] "loc error:  0.4 , std:  3.50180328725448"
# [1] "run time:  0.0291098284721375 , std:  0.00391926438487588"
# [1] " < K*:  0  = K*:  91  > K*:  9"

inspect_one_case(Delta = 50, K = 3, p = 100, delta = 1)
# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  2.65 , std:  5.16471126242721"
# [1] "run time:  0.0280116033554077 , std:  0.00285935887784306"
# [1] " < K*:  0  = K*:  86  > K*:  14"

inspect_one_case(Delta = 200, K = 3, p = 100, delta = 0.5)
# [1] "n 800 , p 100 , Delta 200 , K 3 , kappa 158"
# [1] "loc error:  12.55 , std:  22.1447830005492"
# [1] "run time:  0.0978470730781555 , std:  0.020002773156209"
# [1] " < K*:  0  = K*:  77  > K*:  23"
