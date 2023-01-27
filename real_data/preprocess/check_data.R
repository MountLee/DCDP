setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


library('ecp')

data("DJIA")

X = DJIA$market
n = dim(X)[1]
p = dim(X)[2]

for (i in 1:p){
  X[, i] = X[n:1, i]
}


plot(1:n, X[, 1], type = 'l')

for (i in 1:p){
  
}

DJIA$dates[1138:1140]

library(ggplot2)
library(reshape2)

df = data.frame(X)

# df$time <- DJIA$dates[-c(1,2)]
df$ix <- c(1:dim(X)[1])
# drop the earliest two datetime
df$time <- DJIA$dates[-c(1,2)][n:1]

write.csv(df, 'DJIA.csv', row.names = FALSE)



# Use melt to reshape data so values and variables are in separate columns
df.df <- melt(df, measure.vars = paste("V", c(1:29), sep = ''))

ggplot(df.df, aes(x = time, y = value)) +
  geom_line(aes(color = variable)) +
  # facet_grid(variable ~ ., scales = "free_y") +
  facet_grid(variable ~ ., scales = "fixed") +
  ### Suppress the legend since color isn't actually providing any information
  theme(legend.position = "none")




data("ACGH")
ACGH$individual

df_acgh = data.frame(ACGH$data)
colnames(df_acgh) = ACGH$individual

write.csv(df_acgh, 'ACGH.csv', row.names = FALSE)




devtools::install_github("cykbennie/fbi")
library(fbi)

data("fredmd_description")

# fred = fredmd(file = "E:/python_notebook/DCDP/FRED/2000-01.csv", date_start = as.Date("2000-01-01"), date_end = as.Date("2000-02-01"), transform = TRUE)
# fred = fredmd(file = "E:/python_notebook/DCDP/FRED/2000-01.csv", transform = TRUE)
# fred = fredmd(file = "E:/python_notebook/DCDP/FRED/2010-01.csv", transform = TRUE)
# fred = fredmd(file = "E:/python_notebook/DCDP/FRED/2020-01.csv", transform = TRUE)
fred = fredmd(file = "E:/python_notebook/DCDP/FRED/2022-08.csv", transform = TRUE)
fred$date


ix_start = which(fred$date == as.Date("2000-01-01"))
ix_end = which(fred$date == as.Date("2019-12-01"))
ix_end - ix_start

fred = fred[ix_start:ix_end,]
sum(is.na(fred))

fred = rm_outliers.fredmd(fred)
sum(is.na(fred))

write.csv(fred, 'FRED.csv', row.names = FALSE)



new_df <- fred[ , colSums(is.na(fred))==0]


