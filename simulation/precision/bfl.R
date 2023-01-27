setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# install.packages("reticulate")
library("reticulate")
# for potential errors, see https://github.com/conda/conda/issues/11795
np <- import("numpy")

install.packages("LinearDetect")
library(LinearDetect)
source("cpd_utils.R")

#### load data ####

get_covariance <- function(delta_1, delta_2, p){
  covv = diag(rep(delta_1, p))
  for (i in 1:(p - 1)){
    covv[i, i + 1] = delta_2
    covv[i + 1, i] = delta_2
  }
  return(covv)
}


##### p = 10, n = 400 #####


K = 3
TT = K + 1
Delta = 100
n = Delta * TT
p = 10

delta_1 = 5
delta_2 = 0.3
theta = array(0, dim = c(TT, p, p))

theta[1, , ] = diag(rep(1, p))
theta[2, , ] = get_covariance(delta_1, delta_2, p)
theta[3, , ] = diag(rep(1, p))
theta[4, , ] = get_covariance(delta_1, delta_2, p)

kappa = sum((theta[1, , ] - theta[2, , ])**2)**0.5

data = np$load(paste("data_n", n, "_p", p, "_Delta", Delta, "_K", K, "_kappa", floor(kappa * 100), '.npz', sep = ''))


data$files
Y_train_list = data$f[['Y_train_list']]
Y_test_list = data$f[['Y_test_list']]
cp_truth_list = data$f[['cp_truth_list']]


#### block fused lasso ####

#### b = 1


set.seed(0)
b = 1

Y_train = Y_train_list[b, , ]
Y_test = Y_test_list[b, , ]
n = dim(Y_train)[1]
p = dim(Y_train)[2]
Y_all = array(0, dim = c(2 * n, p))
Y_all[seq(1, 2 * n, 2), ] = Y_train
Y_all[seq(2, 2 * n, 2), ] = Y_test
#### the data_y in bfl should be n-by-p
start_time <- Sys.time()
sink("bfl_log.txt")
cp_estimate = tryCatch(
  expr = {
    temp <- tbfl(method = "GGM", data_y = Y_all,
                 block.size = 2)
    round(temp$cp.final / 2)
  },
  error = function(cond){return (c())}
)
sink()

run_time = difftime(Sys.time(), start_time, units = "secs")
cp_distance(c(0, cp_estimate, n), c(0, cp_truth_list[b, ], n))
run_time

# > cp_distance(c(0, cp_estimate, n), c(0, cp_truth_list[b, ], n))
# [1] 198
# > run_time
# Time difference of 18.45407 secs






#### B = 100

set.seed(0)
# determine the threshold

B = 100

cp_estimate_list = c()
loc_error_list = rep(Inf, B)
K_list = rep(0, B)
run_time_list = rep(NA, B)

for (b in 1:B){
  Y_train = Y_train_list[b, , ]
  Y_test = Y_test_list[b, , ]
  n = dim(Y_train)[1]
  p = dim(Y_train)[2]
  Y_all = array(0, dim = c(2 * n, p))
  Y_all[seq(1, 2 * n, 2), ] = Y_train
  Y_all[seq(2, 2 * n, 2), ] = Y_test
  #### the data_y in bfl should be n-by-p
  start_time <- Sys.time()
  sink("bfl_log.txt", append=TRUE)
  cp_estimate = tryCatch(
    expr = {
      temp <- tbfl(method = "GGM", data_y = Y_all,
                   block.size = 2)
      round(temp$cp.final / 2)
    },
    error = function(cond){return (c())}
  )
  sink()
  run_time = difftime(Sys.time(), start_time, units = "secs")
  cp_estimate_list[[b]] = cp_estimate
  loc_error_list[b] = cp_distance(c(0, cp_estimate, n), c(0, cp_truth_list[b, ], n))
  K_list[b] = length(cp_estimate)
  run_time_list[b] = run_time
}
sink()

{
  print(paste("n", n, ", p", p, ", Delta", Delta, ", K", K, ", kappa", floor(kappa * 100)))
  print(paste("loc error: ", mean(loc_error_list), ", std: ", var(loc_error_list)^0.5))
  print(paste("run time: ", mean(run_time_list), ", std: ", var(run_time_list)^0.5))
  print(paste(" < K*: ", sum(K_list < K), " = K*: ", sum(K_list == K), " > K*: ", sum(K_list > K)))
}








#### more cases ####

bfl_one_case <- function(Delta, K, p, delta_1, delta_2){
  
  TT = K + 1
  n = Delta * TT
  
  theta = array(0, dim = c(TT, p, p))
  
  theta[1, , ] = diag(rep(1, p))
  theta[2, , ] = get_covariance(delta_1, delta_2, p)
  theta[3, , ] = diag(rep(1, p))
  theta[4, , ] = get_covariance(delta_1, delta_2, p)
  
  kappa = sum((theta[1, , ] - theta[2, , ])**2)**0.5
  
  data = np$load(paste("data_n", n, "_p", p, "_Delta", Delta, "_K", K, "_kappa", floor(kappa * 100), '.npz', sep = ''))
  
  Y_train_list = data$f[['Y_train_list']]
  Y_test_list = data$f[['Y_test_list']]
  cp_truth_list = data$f[['cp_truth_list']]
  theta = data$f[['theta']]
  
  
  
  #### projected WBS ####
  
  
  set.seed(0)
  # determine the threshold
  
  B = 100
  
  cp_estimate_list = c()
  loc_error_list = rep(Inf, B)
  K_list = rep(0, B)
  run_time_list = rep(NA, B)
  
  for (b in 1:B){
    Y_train = Y_train_list[b, , ]
    Y_test = Y_test_list[b, , ]
    n = dim(Y_train)[1]
    p = dim(Y_train)[2]
    Y_all = array(0, dim = c(2 * n, p))
    Y_all[seq(1, 2 * n, 2), ] = Y_train
    Y_all[seq(2, 2 * n, 2), ] = Y_test
    #### the data_y in bfl should be n-by-p
    start_time <- Sys.time()
    sink("bfl_log.txt", append=TRUE)
    cp_estimate = tryCatch(
      expr = {
        temp <- tbfl(method = "GGM", data_y = Y_all,
                     block.size = 2)
        round(temp$cp.final / 2)
      },
      error = function(cond){return (c())}
    )
    sink()
    run_time = difftime(Sys.time(), start_time, units = "secs")
    cp_estimate_list[[b]] = cp_estimate
    loc_error_list[b] = cp_distance(c(0, cp_estimate, n), c(0, cp_truth_list[b, ], n))
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


#### Delta = 100
pwbs_one_case(Delta = 100, K = 3, p = 10, delta_1 = 5, delta_2 = 0.3)

pwbs_one_case(Delta = 100, K = 3, p = 20, delta_1 = 5, delta_2 = 0.3)




























##### large sample case #####

K = 3
TT = K + 1
Delta = 500
n = Delta * TT
p = 10

delta_1 = 5
delta_2 = 0.3
theta = array(0, dim = c(TT, p, p))

theta[1, , ] = diag(rep(1, p))
theta[2, , ] = get_covariance(delta_1, delta_2, p)
theta[3, , ] = diag(rep(1, p))
theta[4, , ] = get_covariance(delta_1, delta_2, p)

kappa = sum((theta[1, , ] - theta[2, , ])**2)**0.5

data = np$load(paste("data_n", n, "_p", p, "_Delta", Delta, "_K", K, "_kappa", floor(kappa * 100), '.npz', sep = ''))


data$files
Y_train_list = data$f[['Y_train_list']]
Y_test_list = data$f[['Y_test_list']]
cp_truth_list = data$f[['cp_truth_list']]



#### b = 1


set.seed(0)
b = 1

Y_train = Y_train_list[b, , ]
Y_test = Y_test_list[b, , ]
n = dim(Y_train)[1]
p = dim(Y_train)[2]
Y_all = array(0, dim = c(2 * n, p))
Y_all[seq(1, 2 * n, 2), ] = Y_train
Y_all[seq(2, 2 * n, 2), ] = Y_test
#### the data_y in bfl should be n-by-p
start_time <- Sys.time()
sink("bfl_log.txt")
cp_estimate = tryCatch(
  expr = {
    temp <- tbfl(method = "GGM", data_y = Y_all,lambda.1.cv = 0.1,lambda.1.cv = 0.1,
                 block.size = 100)
    min(temp$cp.final)
  },
  error = function(cond){return (c())}
)
sink()

run_time = difftime(Sys.time(), start_time, units = "secs")
cp_distance(c(0, cp_estimate, n), c(0, cp_truth_list[b, ], n))
run_time

# > cp_distance(c(0, cp_estimate, n), c(0, cp_truth_list[b, ], n))
# [1] 1654
# > run_time
# Time difference of 227.3363 secs

#### too slow





#### p = 100

pwbs_one_case(Delta = 500, K = 3, p = 5, delta_1 = 2, delta_2 = 0.3)

pwbs_one_case(Delta = 500, K = 3, p = 10, delta_1 = 5, delta_2 = 0.3)

pwbs_one_case(Delta = 500, K = 3, p = 20, delta_1 = 5, delta_2 = 0.3)
