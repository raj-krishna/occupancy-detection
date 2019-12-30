#############################################
# Installing or loading necessary libraries #
#############################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(mlbench)) install.packages("mlbench", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(wordcloud)) install.packages("wordcloud", repos = "http://cran.us.r-project.org")
if(!require(DataExplorer)) install.packages("DataExplorer", repos = "http://cran.us.r-project.org")
if(!require(psych)) install.packages("psych", repos = "http://cran.us.r-project.org")
if(!require(mda)) install.packages("mda", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(C50)) install.packages("C50", repos = "http://cran.us.r-project.org")
if(!require(fda)) install.packages("earth", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")



################################################################################################
# Occupancy data:                                                                              # 
# https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip           #
# This zip contains data in 3 files,  namely datatraining.txt, datatest.txt, and datatest2.txt #
# The below code downloads the zip files, reads the 3 files and combines it to deliver         #
# a single dataset of 20,560 observation and 7 variables.                                      #
# An extraneous columns, apart from the 7 named observations, exists in the file, which        #
# I am dropping as it's not pertinent to our project. While I am retaining date, for the       #
# purposes of this project, I would not be using it. It could be taken up as part of           #
# further analysis or improvements.                                                            #
################################################################################################

#############################################
# Data setup                                #
#############################################

# Downloading, reading and combining files to create the dataset
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip", dl)
dl

# List of file names in the zip file
files <- c("datatraining.txt", "datatest.txt", "datatest2.txt")

# Column names for the data in the file
colnames <- c("Date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy")

# Function definition to read data and drop extraneous first column which is an index
read_and_combine_files <- function(file, dl, colnames) {
  data <- fread(text = gsub(",", "\t", readLines(unzip(dl, file))),
                drop = 1, col.names = colnames)
  return(data)
}

# Combining the data from the files into a dataframe
occupancy_detection_data <- bind_rows(lapply(files, read_and_combine_files, dl, colnames))

# Removing variables that are no longer necessary
remove(dl, files, colnames)

#############################################
# Exploratory Data Analysis                 #
#############################################

# List the structure of the occupancy detectinon data
glimpse(occupancy_detection_data)

# Converting occupancy to factor
occupancy_detection_data$Occupancy <- as.factor(occupancy_detection_data$Occupancy)

# Renaming the levels for occupancy as '0' & '1' are not valid variable names in R and 
# would cause caretList method to fail
levels(occupancy_detection_data$Occupancy) <- c("Not_Occupied", "Occupied")

# Formating the Date data
occupancy_detection_data$Date <- as.POSIXct(occupancy_detection_data$Date, tz = "UTC")

# Extracting day of the week from date
occupancy_detection_data$Weekday <- wday(occupancy_detection_data$Date)

# Rearranging the dataframe
occupancy_detection_data <- occupancy_detection_data %>%
  select(Date, Weekday, everything())

# Check the conversions and changes have worked
glimpse(occupancy_detection_data)

# List the variables in the occupancy detection data
#names(occupancy_detection_data)

# Basic statistics of the datasets

introduce(occupancy_detection_data) %>%
  mutate(memory_usage = paste0(round(memory_usage/ 2 ^ 20, 1), " Mb")) %>%
  rename(Rows = rows, Columns = columns,
         "Discrete Columns" = discrete_columns, 
         "Continous Columns" = continuous_columns,
         "All missing columns" = all_missing_columns,
         "Total Missing Values" = total_missing_values,
         "Complete Rows" = complete_rows,
         "Total Observations" = total_observations,
         "Memory Usage" = memory_usage) %>%
  gather() %>% rename(Name = key, Value = value) %>%
  knitr::kable()

# Percentages
plot_intro(occupancy_detection_data)

# Data structure
plot_str(occupancy_detection_data)

# Summary of the data
summary(occupancy_detection_data)

# Missing data profile
plot_missing(occupancy_detection_data)

# Bar Chart by frequency: Occupation 
plot_bar(occupancy_detection_data)

## QQ plot
occupancy_detection_data %>%
  plot_qq()

# Correlation Analysis
occupancy_detection_data %>% select(-Date,-Occupancy) %>%
plot_correlation(type = "all")

################################################################################################
# We can see from the correlation matrix that Humidity and HumidityRatio are highly correlated.#
# Light and Temprature also have significant correlation.                                      #
################################################################################################


################################################################################################
# We will now use Recursive Feature Elimination for feature selection. The following code      #
# would idnetify the optimum number of features that would maximize the accuracy               #
################################################################################################

# Define control for RFE, we are using Random Forest function for selection
rfe_control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)

# RFE algorithm
rfe_results <- rfe(occupancy_detection_data[, 2:7], occupancy_detection_data[[8]],
                   sizes = c(2:7), rfeControl = rfe_control)

# Summarize the results
print(rfe_results)

################################################################################################
# The RFE has selected Light, Temperature, CO2, Weekday, Humidity as the top features. This    #
# would make sense since we have already seen from correlation analysis that Humidity and      #
# Humidity ratio are highly correlated. Using all 6 features does bring a small improvement    #
# in the accuracy. While RFE recommends this, We'll be dropping HumidityRatio for further      #
# processing.                                                                                  #
################################################################################################

predictors(rfe_results)

# Plotting the results
plot(rfe_results, type = c("g", "o"))

# Splitting the data set for training
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(occupancy_detection_data$Occupancy,
                                  times = 1, p = 0.2, list = FALSE)
occupancy_train <- occupancy_detection_data[-test_index,]
occupancy_test <- occupancy_detection_data[test_index,]

# remove text index
rm(test_index)

# Verify the split
glimpse(occupancy_train)
glimpse(occupancy_test)

# Plot the split data
plot_bar(occupancy_train)
plot_bar(occupancy_test)

# Create list of candidate models
models <- c("glm", "lda", "naive_bayes", "svmLinear", 
            "knn", "gamLoess", "multinom", "qda", "mda",
            "rpart", "rf", "C5.0", "fda", "pda", "gbm")

# Run algorithms using 10-fold cross validation
control <-  trainControl(method = "repeatedcv", number = 10, repeats = 3,
                         savePredictions = "final", classProbs = TRUE)
preProcess = c("center", "scale")
# metric <- "Accuracy"

# Training models while suppressing the in function messages
invisible(capture.output(fits <- lapply(models, function(model){
  print(paste0("Now training ", model, " model ..."))
  train(Occupancy ~ .-Date-HumidityRatio, method = model, data = occupancy_train, trControl = control,
        preProc = preProcess)
})))

# Resampling the results
results <- resamples(list("Generalized Linear Model" = fits[[1]], 
                          "Linear Discriminant Analysis" = fits[[2]],
                          "Naive Bayes" = fits[[3]], 
                          "Support Vector Machines with Linear Kernel" = fits[[4]],
                          "k-Nearest Neighbors" = fits[[5]], 
                          "Generalized Additive Model using LOESS" = fits[[6]],
                          "Penalized Multinomial Regression" = fits[[7]], 
                          "Quadratic Discriminant Analysis" = fits[[8]], 
                          "Mixture Discriminant Analysis" = fits[[9]],
                          "CART" = fits[[10]], 
                          "Random Forest" = fits[[11]],
                          "C5.0" = fits[[12]], 
                          "Flexible Discriminant Analysis" = fits[[13]],
                          "Penalized Discriminant Analysis" = fits[[14]],
                          "Stochastic Gradient Boosting" = fits[[15]]),
                     decreasing = TRUE)

# Check the model accuracy
dotplot(results)

# Accuracy with box whisker plots
bwplot(results)

################################################################################################
# We can see that "C5.0" is the top model for this classification model based on accuracy.     #
# Let's now predict the occupancy in the test set to see how "C5.0" works in generalization by #
# making prediction and checking the accuracy.                                                 #
################################################################################################
accuracies <- c()

# Identifying maximum accuracies for each fit
for(ind in 1:length(fits)) {
  accuracies[ind] <- max(fits[[ind]]$results["Accuracy"]) #== "C5.0") {best_model_index = ind}
}

# Identifying the index with the maximum accuracy
best_model_index <- which.max(accuracies)
best_model_name <- fits[[best_model_index]]$method

# Name of the best model
best_model_name

# Assign best model fit
best_model <- fits[[best_model_index]]

# Make predictions
occupancy_preds <- predict(best_model, occupancy_test)

# Final accuracy from the best model in our list of models
confusionMatrix(as.factor(occupancy_preds), occupancy_test$Occupancy)$overall["Accuracy"]

