setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# install.packages("reticulate")
library("reticulate")
# for potential errors, see https://github.com/conda/conda/issues/11795
np <- import("numpy")

library(glmnet)
library(gglasso)
source("cpd_utils.R")

source("VPBS-main/functions.R")
source("VPBS-main/vpcusum.R")
source("VPBS-main/cv-vpcusum.R")
library("MASS")

#### load data ####

K = 3
TT = K + 1
Delta = 50
n = Delta * TT
p = 100

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



#### projected WBS ####


set.seed(0)

B = 10
M = 100
N = 2 * n

delta = 20

vp_tau_list = c(30)
lambda_list = c(0.1)

b = 1
Y_train = Y_train_list[b, , ]
X_train = t(X_train_list[b, , ])
Y_test = Y_test_list[b, , ]
X_test = t(X_test_list[b, , ])


start_time = Sys.time()
random_intervals = gen.intervals(n, M)
vp_estimate = cv.vpcusum(random_intervals, Y_train, X_train, delta ,lambda_list, vp_tau_list, Y_test, X_test)
run_time = difftime(Sys.time(), start_time, units = "secs")
print(run_time)
cp_estimate = vp_estimate[-c(1,length(vp_estimate))]
cp_distance(cp_estimate, cp_truth_list[b, ])





cp_estimate_list = c()
loc_error_list = rep(Inf, B)
K_list = rep(0, B)
run_time_list = rep(NA, B)


for (b in 1:B){
  Y_train = Y_train_list[b, , ]
  X_train = t(X_train_list[b, , ])
  Y_test = Y_test_list[b, , ]
  X_test = t(X_test_list[b, , ])
  #### the x in inspect should be p-by-n
  start.time <- Sys.time()
  random_intervals = gen.intervals(n, M)
  
  sink("pwbs_log.txt")
  vp_estimate = cv.vpcusum(random_intervals, Y_train, X_train, delta ,lambda_list, vp_tau_list, Y_test, X_test)
  sink()
  
  run_time = difftime(Sys.time(), start_time, units = "secs")
  
  loc_error_list[b] = cp_distance(vp_estimate, c(0, cp_truth_list[b, ], n))
  
  cp_estimate = vp_estimate[-c(1,length(vp_estimate))]
  cp_estimate_list[[b]] = cp_estimate
  K_list[b] = length(cp_estimate)
  run_time_list[b] = run_time
}


{
  print(paste("n", n, ", p", p, ", Delta", Delta, ", K", K, ", kappa", floor(kappa * 100)))
  print(paste("loc error: ", mean(loc_error_list), ", std: ", var(loc_error_list)^0.5))
  print(paste("run time: ", mean(run_time_list), ", std: ", var(run_time_list)^0.5))
  print(paste(" < K*: ", sum(K_list < K), " = K*: ", sum(K_list == K), " > K*: ", sum(K_list > K)))
}





#### more cases ####

pwbs_one_case <- function(Delta, K, p, delta_signal, delta_vp = 5, B_MC = 100, vp_tau_list = c(10)){
  
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
  
  lambda_list = c(0.1)
  
  cp_estimate_list = c()
  loc_error_list = rep(Inf, B_MC)
  K_list = rep(0, B_MC)
  run_time_list = rep(NA, B_MC)
  
  M = 100
  N = 2 * n
  
  for (b in 1:B_MC){
    Y_train = Y_train_list[b, , ]
    X_train = t(X_train_list[b, , ])
    Y_test = Y_test_list[b, , ]
    X_test = t(X_test_list[b, , ])
    #### the x in inspect should be p-by-n
    start_time <- Sys.time()
    random_intervals = gen.intervals(n, M)
    
    sink("pwbs_log.txt")
    vp_estimate = cv.vpcusum(random_intervals, Y_train, X_train, delta_vp, lambda_list, vp_tau_list, Y_test, X_test)
    sink()
    
    run_time = difftime(Sys.time(), start_time, units = "secs")
    
    loc_error_list[b] = cp_distance(vp_estimate, c(0, cp_truth_list[b, ], n))
    
    cp_estimate = vp_estimate[-c(1,length(vp_estimate))]
    cp_estimate_list[[b]] = cp_estimate
    K_list[b] = length(cp_estimate)
    run_time_list[b] = run_time
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
K = 3
TT = K + 1
Delta = 50
n = Delta * TT
p = 20

res = pwbs_one_case(Delta = 50, K = 3, p = 20, delta_signal = 5, delta_vp = 5, B_MC = 100, vp_tau_list = c(30))
{
  print(paste("loc error: ", mean(result$loc_error_list), ", std: ", var(result$loc_error_list)^0.5))
  print(paste("run time: ", mean(result$run_time_list), ", std: ", var(result$run_time_list)^0.5))
  print(paste(" < K*: ", sum(result$K_list < K), " = K*: ", sum(result$K_list == K), " > K*: ", sum(result$K_list > K)))
}




res = pwbs_one_case(Delta = 50, K = 3, p = 20, delta_signal = 1, delta_vp = 5, B_MC = 10, vp_tau_list = c(10))

res = pwbs_one_case(Delta = 50, K = 3, p = 20, delta_signal = 1, delta_vp = 5, B_MC = 10, vp_tau_list = c(20))

res = pwbs_one_case(Delta = 50, K = 3, p = 20, delta_signal = 1, delta_vp = 5, B_MC = 10, vp_tau_list = c(30))

#### p = 100

K = 3
TT = K + 1
Delta = 50
n = Delta * TT
p = 100
N = 2 * n

res = pwbs_one_case(Delta = 50, K = 3, p = 100, delta_signal = 5, delta_vp = 5, B_MC = 100, vp_tau_list = c(30))
sink()
{
  print(paste("loc error: ", mean(res$loc_error_list), ", std: ", var(res$loc_error_list)^0.5))
  print(paste("run time: ", mean(res$run_time_list), ", std: ", var(res$run_time_list)^0.5))
  print(paste(" < K*: ", sum(res$K_list < K), " = K*: ", sum(res$K_list == K), " > K*: ", sum(res$K_list > K)))
}

delta = 5
theta = matrix(0, nrow = TT, ncol = p)
for (k in 1:TT){
  theta[k, (1 + 5 * (k - 1)):(5 * k)] = delta
}
kappa = sum((theta[1,] - theta[2,])**2)**0.5

save(n, p, Delta, delta, res, file=
       paste("n", n, "_p", p, "_Delta", Delta, "_K", K, "_kappa", floor(kappa * 100), ".RData", sep = ""))



res = pwbs_one_case(Delta = 50, K = 3, p = 100, delta_signal = 1, delta_vp = 5, B_MC = 10, vp_tau_list = c(10))


res = pwbs_one_case(Delta = 50, K = 3, p = 100, delta_signal = 1, delta_vp = 5, B_MC = 100, vp_tau_list = c(10))



res = pwbs_one_case(Delta = 50, K = 3, p = 100, delta_signal = 1, delta_vp = 5, B_MC = 100, vp_tau_list = c(5))
# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  11.54 , std:  11.2262499022909"
# [1] "run time:  120.386535067558 , std:  14.4822515055225"
# [1] " < K*:  3  = K*:  65  > K*:  32"




# > res = pwbs_one_case(Delta = 50, K = 3, p = 100, delta = 1, B_MC = 100, vp_tau_list = c(30))
# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  Inf , std:  NaN"
# [1] "run time:  2.12663562401136 , std:  0.186830906546452"
# [1] " < K*:  100  = K*:  0  > K*:  0"

# > res = pwbs_one_case(Delta = 50, K = 3, p = 100, delta = 1, B_MC = 100, vp_tau_list = c(10))
# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  Inf , std:  NaN"
# [1] "run time:  2.07982047375043 , std:  0.178874867622907"
# [1] " < K*:  58  = K*:  37  > K*:  5"

# res = pwbs_one_case(Delta = 50, K = 3, p = 100, delta_signal = 1, delta_vp = 5, B_MC = 100, vp_tau_list = c(10))
# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  24.97 , std:  20.7858027499444"
# [1] "run time:  8533.69786569357 , std:  3376.57141673776"
# [1] " < K*:  59  = K*:  41  > K*:  0"

