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

delta = 1
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

B = 100

cp_estimate_list = c()
loc_error_list = rep(Inf, B)
K_list = rep(0, B)
run_time_list = rep(NA, B)
# b=1
for (b in 1:B){
  Y_train = Y_train_list[b, , ]
  Y_test = Y_test_list[b, , ]
  Y_all = array(0, dim = c(2 * n, p))
  Y_all[seq(1, 2 * n, 2), ] = Y_train
  Y_all[seq(2, 2 * n, 2), ] = Y_test
  #### the x in inspect should be p-by-n
  start_time <- Sys.time()
  
  sink("bfl_log.txt")
  cp_estimate = tryCatch(
    expr = {
      temp <- tbfl(method = "Constant", data_y = Y_all)
      round(temp$cp.final / 2)
    },
    error = function(cond){return (c())}
  )
  sink()

  run_time <- difftime(Sys.time(), start_time, units = "secs")
  
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

# hist(loc_error_list)


# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  43.3 , std:  8.24804843767924"
# [1] "run time:  2.8913914513588 , std:  0.583846351013571"
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
  
  data$files
  Y_train_list = data$f[['Y_train_list']]
  Y_test_list = data$f[['Y_test_list']]
  cp_truth_list = data$f[['cp_truth_list']]
  theta = data$f[['theta']]
  
  
  
  #### projected WBS ####
  
  
  set.seed(0)
  # determine the threshold
  
  # B_MC = 100
  
  cp_estimate_list = c()
  loc_error_list = rep(Inf, B_MC)
  K_list = rep(0, B_MC)
  run_time_list = rep(NA, B_MC)
  
  for (b in 1:B_MC){
    Y_train = Y_train_list[b, , ]
    Y_test = Y_test_list[b, , ]
    Y_all = array(0, dim = c(2 * n, p))
    Y_all[seq(1, 2 * n, 2), ] = Y_train
    Y_all[seq(2, 2 * n, 2), ] = Y_test
    #### the x in inspect should be p-by-n
    start_time <- Sys.time()
    sink("bfl_log.txt")
    cp_estimate = tryCatch(
      expr = {
        temp <- tbfl(method = "Constant", data_y = Y_all)
        round(temp$cp.final / 2)
      },
      error = function(cond){return (c())}
    )
    sink()
    
    run_time <- difftime(Sys.time(), start_time, units = "secs")
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
result = bfl_one_case(Delta = 50, K = 3, p = 20, delta = 5, B_MC = 100)
sink()
{
  print(paste("loc error: ", mean(result$loc_error_list), ", std: ", var(result$loc_error_list)^0.5))
  print(paste("run time: ", mean(result$run_time_list), ", std: ", var(result$run_time_list)^0.5))
  print(paste(" < K*: ", sum(result$K_list < K), " = K*: ", sum(result$K_list == K), " > K*: ", sum(result$K_list > K)))
}

# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 1581"
# [1] "loc error:  42.56 , std:  6.95486168159375"
# [1] "run time:  3.48406048297882 , std:  0.572081911420657"
# [1] " < K*:  100  = K*:  0  > K*:  0"



result = bfl_one_case(Delta = 50, K = 3, p = 20, delta = 1, B_MC = 100)
sink()
{
  print(paste("loc error: ", mean(result$loc_error_list), ", std: ", var(result$loc_error_list)^0.5))
  print(paste("run time: ", mean(result$run_time_list), ", std: ", var(result$run_time_list)^0.5))
  print(paste(" < K*: ", sum(result$K_list < K), " = K*: ", sum(result$K_list == K), " > K*: ", sum(result$K_list > K)))
}




result = bfl_one_case(Delta = 50, K = 3, p = 20, delta = 0.5, B_MC = 100)
sink()
{
  print(paste("loc error: ", mean(result$loc_error_list), ", std: ", var(result$loc_error_list)^0.5))
  print(paste("run time: ", mean(result$run_time_list), ", std: ", var(result$run_time_list)^0.5))
  print(paste(" < K*: ", sum(result$K_list < K), " = K*: ", sum(result$K_list == K), " > K*: ", sum(result$K_list > K)))
}
# [1] "n 200 , p 20 , Delta 50 , K 3 , kappa 158"
# [1] "loc error:  54.48 , std:  20.9768896405044"
# [1] "run time:  2.84157598257065 , std:  1.13920892747157"
# [1] " < K*:  100  = K*:  0  > K*:  0"

#### p = 100

result = bfl_one_case(Delta = 50, K = 3, p = 100, delta = 5, B_MC = 100)
sink()
{
  print(paste("loc error: ", mean(result$loc_error_list), ", std: ", var(result$loc_error_list)^0.5))
  print(paste("run time: ", mean(result$run_time_list), ", std: ", var(result$run_time_list)^0.5))
  print(paste(" < K*: ", sum(result$K_list < K), " = K*: ", sum(result$K_list == K), " > K*: ", sum(result$K_list > K)))
}
# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 1581"
# [1] "loc error:  47.8 , std:  6.65756955076192"
# [1] "run time:  1.46942430496216 , std:  0.280905529112514"
# [1] " < K*:  100  = K*:  0  > K*:  0"

result = bfl_one_case(Delta = 50, K = 3, p = 100, delta = 1, B_MC = 100)
sink()
{
  print(paste("loc error: ", mean(result$loc_error_list), ", std: ", var(result$loc_error_list)^0.5))
  print(paste("run time: ", mean(result$run_time_list), ", std: ", var(result$run_time_list)^0.5))
  print(paste(" < K*: ", sum(result$K_list < K), " = K*: ", sum(result$K_list == K), " > K*: ", sum(result$K_list > K)))
}
# [1] "n 200 , p 100 , Delta 50 , K 3 , kappa 316"
# [1] "loc error:  47.59 , std:  6.07544320202321"
# [1] "run time:  1.13256215572357 , std:  0.231015366263611"
# [1] " < K*:  100  = K*:  0  > K*:  0"


result = bfl_one_case(Delta = 200, K = 3, p = 100, delta = 0.5, B_MC = 100)
sink()
{
  print(paste("loc error: ", mean(result$loc_error_list), ", std: ", var(result$loc_error_list)^0.5))
  print(paste("run time: ", mean(result$run_time_list), ", std: ", var(result$run_time_list)^0.5))
  print(paste(" < K*: ", sum(result$K_list < K), " = K*: ", sum(result$K_list == K), " > K*: ", sum(result$K_list > K)))
}
# [1] "n 800 , p 100 , Delta 200 , K 3 , kappa 158"
# [1] "loc error:  80.1 , std:  137.327044575048"
# [1] "run time:  15.7161379432678 , std:  3.77763650578316"
# [1] " < K*:  28  = K*:  71  > K*:  1"