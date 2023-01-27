setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# install.packages("reticulate")
library("reticulate")
# for potential errors, see https://github.com/conda/conda/issues/11795
np <- import("numpy")

# install.packages("LinearDetect")
library(LinearDetect)
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
X_train_list = data$f[['X_train_list']]
Y_test_list = data$f[['Y_train_list']]
X_test_list = data$f[['X_train_list']]
cp_truth_list = data$f[['cp_truth_list']]
beta = data$f[['beta']]



#### try b = 1 ####


set.seed(0)
# determine the threshold

b = 1
Y_train = Y_train_list[b, , ]
X_train = X_train_list[b, , ]
Y_test = Y_test_list[b, , ]
X_test = X_test_list[b, , ]

Y_all = array(0, dim = c(2 * n, 1))
Y_all[seq(1, 2 * n, 2), ] = Y_train
Y_all[seq(2, 2 * n, 2), ] = Y_test

X_all = array(0, dim = c(2 * n, p))
X_all[seq(1, 2 * n, 2), ] = X_train
X_all[seq(2, 2 * n, 2), ] = X_test
#### the x in tbfl should be n-by-p
start_time <- Sys.time()
# sink("bfl_log.txt")
cp_estimate = tryCatch(
  expr = {
    temp <- tbfl(method, Y_all, X_all)
    min(temp$cp.final)
  },
  error = function(cond){return (c())}
)
# sink()

run_time <- Sys.time() - start_time

cp_estimate_list[[b]] = cp_estimate
loc_error_list[b] = cp_distance(cp_estimate, cp_truth_list[b, ])
K_list[b] = length(cp_estimate)
run_time_list[b] = run_time





#### BFL ####


set.seed(0)
# determine the threshold

B = 100

cp_estimate_list = c()
loc_error_list = rep(Inf, B)
K_list = rep(0, B)
run_time_list = rep(NA, B)

for (b in 1:B){
  Y_train = Y_train_list[b, , ]
  X_train = X_train_list[b, , ]
  Y_test = Y_test_list[b, , ]
  X_test = X_test_list[b, , ]
  
  Y_all = array(0, dim = c(2 * n, 1))
  Y_all[seq(1, 2 * n, 2), ] = Y_train
  Y_all[seq(2, 2 * n, 2), ] = Y_test
  
  X_all = array(0, dim = c(2 * n, p))
  X_all[seq(1, 2 * n, 2), ] = X_train
  X_all[seq(2, 2 * n, 2), ] = X_test
  #### the x in tbfl should be n-by-p
  start_time <- Sys.time()
  sink("bfl_log.txt")
  cp_estimate = tryCatch(
    expr = {
      temp <- tbfl(method = "MLR", Y_all, X_all)
      min(temp$cp.final)
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


# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 1581"
# [1] "loc error:  84.45 , std:  15.3348978121503"
# [1] "run time:  4.18814157962799 , std:  0.709080129422368"
# [1] " < K*:  100  = K*:  0  > K*:  0"


#### more cases ####

bfl_one_case <- function(Delta, K, p, delta, B_MC = 100){
  
  # K = 3
  TT = K + 1
  # Delta = 50
  n = Delta * TT
  # p = 20
  
  # delta = 5
  theta = matrix(0, nrow = TT, ncol = p)
  for (k in 1:TT){
    theta[k, (1 + 5 * (k - 1)):(5 * k)] = delta
  }
  kappa = sum((theta[1,] - theta[2,])**2)**0.5
  
  data = np$load(paste("data_n", n, "_p", p, "_Delta", Delta, "_K", K, "_kappa", floor(kappa * 100), '.npz', sep = ''))
  
  Y_train_list = data$f[['Y_train_list']]
  X_train_list = data$f[['X_train_list']]
  Y_test_list = data$f[['Y_train_list']]
  X_test_list = data$f[['X_train_list']]
  cp_truth_list = data$f[['cp_truth_list']]
  beta = data$f[['beta']]
  
  
  
  #### BFL ####
  
  set.seed(0)
  
  cp_estimate_list = c()
  loc_error_list = rep(Inf, B_MC)
  K_list = rep(0, B_MC)
  run_time_list = rep(NA, B_MC)
  
  for (b in 1:B_MC){
    Y_train = Y_train_list[b, , ]
    X_train = X_train_list[b, , ]
    Y_test = Y_test_list[b, , ]
    X_test = X_test_list[b, , ]
    
    Y_all = array(0, dim = c(2 * n, 1))
    Y_all[seq(1, 2 * n, 2), ] = Y_train
    Y_all[seq(2, 2 * n, 2), ] = Y_test
    
    X_all = array(0, dim = c(2 * n, p))
    X_all[seq(1, 2 * n, 2), ] = X_train
    X_all[seq(2, 2 * n, 2), ] = X_test
    #### the x in tbfl should be n-by-p
    start_time <- Sys.time()
    sink("bfl_log.txt")
    cp_estimate = tryCatch(
      expr = {
        temp <- tbfl(method = "MLR", Y_all, X_all)
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
  
  return(list(cp_estimate_list = cp_estimate_list, loc_error_list = loc_error_list,
              K_list = K_list, run_time_list = run_time_list))
}


#### p = 20

res = bfl_one_case(Delta = 50, K = 3, p = 20, delta = 1, B_MC = 100)


# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  43.31 , std:  8.8199670170482"
# [1] "run time:  3.11247735977173 , std:  0.807235319515982"
# [1] " < K*:  100  = K*:  0  > K*:  0"

res = bfl_one_case(Delta = 50, K = 3, p = 20, delta = 0.5, B_MC = 100)


# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 158"
# [1] "loc error:  52.37 , std:  18.702700609832"
# [1] "run time:  3.01381983757019 , std:  0.996821708491295"
# [1] " < K*:  100  = K*:  0  > K*:  0"



#### p = 100

res = bfl_one_case(Delta = 50, K = 3, p = 100, delta = 5, B_MC = 100)

# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 1581"
# [1] "loc error:  47.84 , std:  6.6949702211904"
# [1] "run time:  1.37721860408783 , std:  0.247506241757016"
# [1] " < K*:  100  = K*:  0  > K*:  0"


res = bfl_one_case(Delta = 50, K = 3, p = 100, delta = 1, B_MC = 100)


# [1] "loc error:  47.19 , std:  6.48338202053854"
# [1] "run time:  1.0627835059166 , std:  0.217480734052856"
# [1] " < K*:  100  = K*:  0  > K*:  0"


res = bfl_one_case(Delta = 200, K = 3, p = 100, delta = 0.5, B_MC = 100)

# [1] "n 800 , p 100 , Delta 200 , K 3 , kappa 158"
# [1] "loc error:  100.31 , std:  111.972732521283"
# [1] "run time:  20.7635567235947 , std:  3.86782071282088"
# [1] " < K*:  45  = K*:  53  > K*:  2"

