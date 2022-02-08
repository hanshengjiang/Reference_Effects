# set up working directory as the location of this script
# if you are running this in rstudio
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# check if R package "mixtools" is installed
if("mixtools" %in% rownames(installed.packages()) == FALSE) {install.packages("mixtools")}
require("mixtools")

# receive arguments from command line
args <- commandArgs(trailingOnly = TRUE)
# number of components taken from command line arguments
# algorithm tends to 
num_component <- as.numeric(args[1]) 
thresh <- num_component
seed_id <- as.integer(args[2])
folder_path <- args[3]

# set random seed
set.seed(seed_id)

#---------------------------------------------------------------------------#
# read simulated data from file
dir <- file.path(folder_path)
X <- as.matrix(read.csv(file = file.path(dir, "X.csv"), header = FALSE))
y <- c(as.matrix(read.csv(file = file.path(dir, "y.csv"), header = FALSE)))
#---------------------------------------------------------------------------#

if(file.exists(file.path(dir, "B_logit.csv"))) {
  print("---Reading initialization from standard logistic regression---")
B_logit <- as.matrix(read.csv(file = file.path(dir, "B_logit.csv"), header = FALSE))
len <- length(c(B_logit))
B_logit <- as.matrix(c(B_logit), nrow = len)
num_component <- min(thresh, num_component)
beta_ini <- matrix(rep(B_logit, each = num_component), nrow = nrow(B_logit), byrow = TRUE)

#---------------------------------------------------------------------------#
# call logisregmixEM from R Package mixtools
# https://rdrr.io/cran/mixtools/man/logisregmixEM.html
#---------------------------------------------------------------------------#
reg_outcome <- logisregmixEM(
                 y, X, 
                 k = num_component,
                 beta = beta_ini,
                 lambda = rep(1/num_component, as.integer(num_component)), # initialization of mixing proportions
                 addintercept = FALSE, 
                 epsilon = 1e-03,
                 maxit = 50, # max number of iterations
                 verb = TRUE # print iterations
                 )
#---------------------------------------------------------------------------#
} else { # no initialization provided
  #---------------------------------------------------------------------------#
  # call logisregmixEM from R Package mixtools
  # https://rdrr.io/cran/mixtools/man/logisregmixEM.html
  #---------------------------------------------------------------------------#
  reg_outcome <- logisregmixEM(
    y, X, 
    k = num_component,
    addintercept = FALSE, 
    epsilon = 1e-03,
    maxit = 500, # max number of iterations
    verb = FALSE # print iterations
  )
  #---------------------------------------------------------------------------#
}

# storage estimation results to files
alpha_EM <- reg_outcome$lambda
write.table(alpha_EM, file = file.path(dir, "alpha_EM.csv"),sep = ",",row.names = FALSE, col.names  = FALSE)

B_EM <- reg_outcome$beta
write.table(B_EM, file = file.path(dir, "B_EM.csv"),sep = ",",row.names = FALSE, col.names  = FALSE)



