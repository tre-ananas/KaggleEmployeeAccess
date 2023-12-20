#################################################################
#################################################################
# Amazon Employee Access Challenge    ###########################
# Ryan Wolff                          ###########################
# 6 October 2023                      ###########################
#################################################################
#################################################################

#################################################################
#################################################################
# LOAD DATA AND PACKAGES              ###########################
#################################################################
#################################################################

# Data Location and Description
# https://www.kaggle.com/competitions/amazon-employee-access-challenge/data

# Install Packages (For Use on Remote Server)
# install.packages('doParallel')
# install.packages('vroom')
# install.packages('DataExplorer')
# install.packages('patchwork')
# install.packages('inspectdf')
# install.packages('ggmosaic')
# install.packages('tidyverse')
# install.packages('tidymodels')
# install.packages('embed')
# install.packages('lme4')
# install.packages('naivebayes')
# install.packages('discrim')
# install.packages('kknn')
# install.packages('stacks')
# install.packages('kernlab')
# install.packages('themis')

# Load Libraries
# library(doParallel) # Parallel Computing
# library(vroom) # Loading data
# library(DataExplorer) # EDA
# library(patchwork) # EDA
# library(inspectdf) # EDA
# library(ggmosaic) # EDA
# library(tidyverse) # General Use
# library(tidymodels) # General Modeling
# library(embed) # plogr modeling
# library(lme4) # plogr modeling
# library(naivebayes) # Naive Bayes modeling
# library(discrim) # Naive Bayes modeling
# library(kknn) # K nearest neighbors
# library(stacks) # Model Stacking
# library(kernlab) # SVMS
# library(themis) # Balancing Data - SMOTE

########################################################################
########################################################################
# #############################
# # Start run in parallel
# # cl <- makePSOCKcluster(3)
# # registerDoParallel(cl)
# #############################
########################################################################
########################################################################

# # Load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# #################################################################
# #################################################################
# # EDA                                 ###########################
# #################################################################
# #################################################################
# 
# # Load Libraries
# # library(DataExplorer) # EDA
# # library(patchwork) # EDA
# # library(inspectdf) # EDA
# # library(ggmosaic) # EDA
# # library(tidyverse) # General Use
# 
# # # Create an EDA dataset and correct Vroom's mistake and make numeric data into factors
# # fact_employee_train_eda <- employee_train
# # cols <- c("ACTION",
# #           "RESOURCE", 
# #           "MGR_ID", 
# #           "ROLE_ROLLUP_1", 
# #           "ROLE_ROLLUP_2", 
# #           "ROLE_DEPTNAME", 
# #           "ROLE_TITLE",
# #           "ROLE_FAMILY_DESC",
# #           "ROLE_FAMILY",
# #            "ROLE_CODE")
# # fact_employee_train_eda[cols] <- lapply(fact_employee_train_eda[cols], factor)
# 
# # # Examine Factor Variables:
# #   # cnt = # unique variables
# #   # common = most common level
# #   # common_pcnt = percentage representing most common level
# #   # levels = list of the proportions of each level of the variable
# # factor_exploration_plot <- fact_employee_train_eda %>%
# #   inspect_cat() %>%
# #   show_plot()
# # factor_exploration_plot
# # 
# # fact_employee_train_eda %>%
# #   inspect_cat()
# # 
# # # Create an EDA dataset making every feature into a factor except ACTION, which remains numeric
# # num_employee_train_eda <- employee_train
# # cols <- c("RESOURCE", 
# #           "MGR_ID", 
# #           "ROLE_ROLLUP_1", 
# #           "ROLE_ROLLUP_2", 
# #           "ROLE_DEPTNAME", 
# #           "ROLE_TITLE",
# #           "ROLE_FAMILY_DESC",
# #           "ROLE_FAMILY",
# #           "ROLE_CODE")
# # num_employee_train_eda[cols] <- lapply(num_employee_train_eda[cols], 
# #                                    factor)
# # 
# # # Identify the top 30 most popular RESOURCEs
# # resource_data <- num_employee_train_eda %>%
# #   group_by(RESOURCE) %>%
# #   summarize(mean = mean(ACTION),
# #             n = n()) %>%
# #   slice_max(order_by = n,
# #             n = 30)
# # resource_data <- resource_data["RESOURCE"]
# # 
# # # Subset the EDA dataset to the 30 most popular RESOURCEs
# # num_employee_train_eda <- num_employee_train_eda %>%
# #   filter(RESOURCE %in% resource_data$RESOURCE)
# # 
# # # Bar chart of the top 30 most popular RESOURCEs' ACTION results
# # action_resources_barcharts <- ggplot(data = num_employee_train_eda,
# #        aes(x = ACTION)) +
# #   geom_bar() +
# #   ggtitle("ACTION Results for 30 Most Common Products") +
# #   xlab("ACTION: 0 and 1, Respectively") +
# #   ylab("Count of Each ACTION Result") +
# #   theme(plot.title = element_text(hjust = .5)) +
# #   facet_wrap( ~ RESOURCE)
# # action_resources_barcharts
# # 
# # # Create a 2-Way Plot of Prominent Plots
# # twoway_patch <- (factor_exploration_plot) / (action_resources_barcharts)
# # twoway_patch
#   
# # Findings:
#   # inspect_cat():
#     # ACTION is extremely homogenous--94.2% are in value 1, aka the resource was approved
#     # ROLE_ROLLUP_1 is extremely homogenous--65.3% are in value 117961
#   # show_plot() is a visualization of inspect_cat()
#   # geom_bar():
#     # 20897 is the only RESOURCE that is not overwhelmingly approved
# 
# #################################################################
# #################################################################
# # LOGISTIC REGRESSION                 ###########################
# #################################################################
# #################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# # library(tidymodels)
# # library(tidyverse)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# logr_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than 1% of the time into an "other" category
#   step_other(all_nominal_predictors(), threshold = .01) %>%
#   # Dummy variable encoding for all nominal predictors
#   step_dummy(all_nominal_predictors())
# 
# 
# # Prep, Bake, and View Recipe
# logr_prep <- prep(logr_rec)
# bake(logr_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create logistic regression model
# logr_mod <- logistic_reg() %>%
#   set_engine("glm")
# 
# # Create and fit logistic regression workflow
# logr_wf <- workflow() %>%
#   add_recipe(logr_rec) %>%
#   add_model(logr_mod) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Predict with classification cutoff = .70
# logr_preds <- predict(logr_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   mutate(ifelse(.pred_1 > .83, 1, 0)) %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = names(.)[4]) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# # vroom_write(x=logr_preds, file="logr_preds.csv", delim = ",")
# 
# # Predict without a classification cutoff--just the raw probabilities
# logr_preds_no_c <- predict(logr_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=logr_preds_no_c, file="logr_preds_no_c.csv", delim = ",")
# 
# #################################################################
# #################################################################
# # PENALIZED LOGISTIC REGRESSION       ###########################
# #################################################################
# #################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# # library(tidymodels)
# # library(tidyverse)
# # library(embed)
# # library(lme4)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# plogr_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than .1% of the time into an "other" category
#   # Remove because penalized logr can handle categories w few observations
#   # step_other(all_nominal_predictors(), threshold = .001) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# 
# # Prep, Bake, and View Recipe
# plogr_prep <- prep(plogr_rec)
# bake(plogr_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create penalized logistic regression model
# plogr_mod <- logistic_reg(mixture = tune(),
#                           penalty = tune()) %>%
#   set_engine("glmnet")
# 
# # Create logistic regression workflow
# plogr_wf <- workflow() %>%
#   add_recipe(plogr_rec) %>%
#   add_model(plogr_mod)
# 
# # Grid of values to tune over
# plogr_tg <- grid_regular(penalty(),
#                          mixture(),
#                          levels = 5)
# 
# # Split data for cross-validation (CV)
# plogr_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# plogr_cv_results <- plogr_wf %>%
#   tune_grid(resamples = plogr_folds,
#             grid = plogr_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# plogr_best_tune <- plogr_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# plogr_final_wf <- plogr_wf %>%
#   finalize_workflow(plogr_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Predict without a classification cutoff--just the raw probabilities
# plogr_preds_no_c <- predict(plogr_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=plogr_preds_no_c, file="plogr_preds_no_c_no_step_other.csv", delim = ",")
# 
# #################################################################
# #################################################################
# # CLASSIFICATION FOREST               ###########################
# #################################################################
# #################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# # library(tidymodels)
# # library(tidyverse)
# # library(embed)
# # library(lme4)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# ctree_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than .1% of the time into an "other" category
#   # Removed to see if classification trees can handle smaller data groups
#   # step_other(all_nominal_predictors(), threshold = .001) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# 
# # Prep, Bake, and View Recipe
# ctree_prep <- prep(ctree_rec)
# bake(ctree_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create classification forest model
# ctree_mod <- rand_forest(mtry = tune(),
#                          min_n = tune(),
#                          trees = 500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create classification forest workflow
# ctree_wf <- workflow() %>%
#   add_recipe(ctree_rec) %>%
#   add_model(ctree_mod)
# 
# # Grid of values to tune over
# ctree_tg <- grid_regular(mtry(range = c(1, 9)),
#                          min_n(),
#                          levels = 5)
# 
# # Split data for cross-validation (CV)
# ctree_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# ctree_cv_results <- ctree_wf %>%
#   tune_grid(resamples = ctree_folds,
#             grid = ctree_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# ctree_best_tune <- ctree_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# ctree_final_wf <- ctree_wf %>%
#   finalize_workflow(ctree_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Predict without a classification cutoff--just the raw probabilities
# ctree_preds <- predict(ctree_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=ctree_preds, file="ctree_preds.csv", delim = ",")
# 
# #################################################################
# #################################################################
# # random fore BAYES                         ###########################
# #################################################################
# #################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# # library(tidymodels)
# # library(tidyverse)
# # library(naivebayes)
# # library(discrim)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# nb_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than .1% of the time into an "other" category
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# 
# # Prep, Bake, and View Recipe
# nb_prep <- prep(nb_rec)
# bake(nb_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create Naive Bayes model
# nb_mod <- naive_Bayes(Laplace = tune(),
#                       smoothness = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# # Create Naive Bayes workflow
# nb_wf <- workflow() %>%
#   add_recipe(nb_rec) %>%
#   add_model(nb_mod)
# 
# # Grid of values to tune over
# nb_tg <- grid_regular(Laplace(),
#                       smoothness(),
#                       levels = 10)
# 
# # Split data for cross-validation (CV)
# nb_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# nb_cv_results <- nb_wf %>%
#   tune_grid(resamples = nb_folds,
#             grid = nb_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# nb_best_tune <- nb_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# nb_final_wf <- nb_wf %>%
#   finalize_workflow(nb_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Make predictions
# nb_preds <- predict(nb_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=nb_preds, file="nb_preds.csv", delim = ",")
# 
# #################################################################
# #################################################################
# # K Nearest Neighbors                 ###########################
# #################################################################
# #################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# # library(kknn)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# knn_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than .1% of the time into an "other" category
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   # Normalize Numeric Predictors
#   step_normalize(all_numeric_predictors())
# 
# 
# # Prep, Bake, and View Recipe
# knn_prep <- prep(knn_rec)
# bake(knn_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create K Nearest Neighbors model
# knn_mod <- nearest_neighbor(neighbors = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# # Create KNN workflow
# knn_wf <- workflow() %>%
#   add_recipe(knn_rec) %>%
#   add_model(knn_mod)
# 
# # Grid of values to tune over
# knn_tg <- grid_regular(neighbors(),
#                       levels = 10)
# 
# # Split data for cross-validation (CV)
# knn_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# knn_cv_results <- knn_wf %>%
#   tune_grid(resamples = knn_folds,
#             grid = knn_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# knn_best_tune <- knn_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# knn_final_wf <- knn_wf %>%
#   finalize_workflow(knn_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Make predictions
# knn_preds <- predict(knn_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=knn_preds, file="knn_preds.csv", delim = ",")
# 
# #################################################################
# #################################################################
# # NAIVE BAYES WITH PRINCIPAL COMPONENT REDUCTION ################
# #################################################################
# #################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# # library(tidymodels)
# # library(tidyverse)
# # library(naivebayes)
# # library(discrim)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# nbpcr_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than .1% of the time into an "other" category
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   # Normalize
#   step_normalize(all_predictors()) %>%
#   # PCS Threshold = .92
#   step_pca(all_predictors(), threshold = .85)
# 
# 
# # Prep, Bake, and View Recipe
# nbpcr_prep <- prep(nbpcr_rec)
# bake(nbpcr_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create Naive Bayes model
# nbpcr_mod <- naive_Bayes(Laplace = tune(),
#                       smoothness = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# # Create Naive Bayes workflow
# nbpcr_wf <- workflow() %>%
#   add_recipe(nbpcr_rec) %>%
#   add_model(nbpcr_mod)
# 
# # Grid of values to tune over
# nbpcr_tg <- grid_regular(Laplace(),
#                       smoothness(),
#                       levels = 10)
# 
# # Split data for cross-validation (CV)
# nbpcr_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# nbpcr_cv_results <- nbpcr_wf %>%
#   tune_grid(resamples = nbpcr_folds,
#             grid = nbpcr_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# nbpcr_best_tune <- nbpcr_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# nbpcr_final_wf <- nbpcr_wf %>%
#   finalize_workflow(nbpcr_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Make predictions
# nbpcr_preds <- predict(nbpcr_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=nbpcr_preds, file="nbpcr85_preds.csv", delim = ",")
# 
# #################################################################
# #################################################################
# # K Nearest Neighbors With Principal Component Reduction ########
# #################################################################
# #################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# # library(kknn)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# knnpcr_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than .1% of the time into an "other" category
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   # Normalize Numeric Predictors
#   step_normalize(all_numeric_predictors()) %>%
#   # PCA w/ Threshold = .90
#   step_pca(all_predictors(), threshold = .90)
# 
# 
# # Prep, Bake, and View Recipe
# knnpcr_prep <- prep(knnpcr_rec)
# bake(knnpcr_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create K Nearest Neighbors model
# knnpcr_mod <- nearest_neighbor(neighbors = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# # Create KNN workflow
# knnpcr_wf <- workflow() %>%
#   add_recipe(knnpcr_rec) %>%
#   add_model(knnpcr_mod)
# 
# # Grid of values to tune over
# knnpcr_tg <- grid_regular(neighbors(),
#                       levels = 10)
# 
# # Split data for cross-validation (CV)
# knnpcr_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# knnpcr_cv_results <- knnpcr_wf %>%
#   tune_grid(resamples = knnpcr_folds,
#             grid = knnpcr_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# knnpcr_best_tune <- knnpcr_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# knnprc_final_wf <- knnpcr_wf %>%
#   finalize_workflow(knnpcr_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Make predictions
# knnpcr_preds <- predict(knnprc_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=knnpcr_preds, file="knnpcr90_preds.csv", delim = ",")

#################################################################
#################################################################
# CLASSIFICATION FOREST WITH N LEVELS OF CROSS VALIDATION ######
#################################################################
#################################################################

# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# library(vroom)
# library(tidymodels)
# library(tidyverse)
# library(embed)
# library(lme4)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# ctree_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
# 
# # Prep, Bake, and View Recipe
# ctree_prep <- prep(ctree_rec)
# bake(ctree_prep, employee_train)
# 
# # MODELING ------------------------------------------------------
# 
# # Create classification forest model
# ctree_mod <- rand_forest(mtry = tune(),
#                          min_n = tune(),
#                          trees = 750) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create classification forest workflow
# ctree_wf <- workflow() %>%
#   add_recipe(ctree_rec) %>%
#   add_model(ctree_mod)
# 
# # Grid of values to tune over
# ctree_tg <- grid_regular(mtry(range = c(1, 9)),
#                          min_n(),
#                          levels = 15)
# 
# # Split data for cross-validation (CV)
# ctree_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# ctree_cv_results <- ctree_wf %>%
#   tune_grid(resamples = ctree_folds,
#             grid = ctree_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# ctree_best_tune <- ctree_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# ctree_final_wf <- ctree_wf %>%
#   finalize_workflow(ctree_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Predict without a classification cutoff--just the raw probabilities
# ctree_preds <- predict(ctree_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=ctree_preds, file="ctree_preds_15levels.csv", delim = ",")

#################################################################
#################################################################
# CLASSIFICATION FOREST WITH N LEVELS OF CROSS VALIDATION AND PCR
#################################################################
#################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# library(vroom)
# library(tidymodels)
# library(tidyverse)
# library(embed)
# library(lme4)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# ctreepcr_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   # Normalize
#   step_normalize(all_predictors()) %>%
#   # PCR w/ Threshold = .9
#   step_pca(all_predictors(), threshold = .9)
# 
# # Prep, Bake, and View Recipe
# ctreepcr_prep <- prep(ctreepcr_rec)
# bake(ctreepcr_prep, employee_train)
# 
# # MODELING ------------------------------------------------------
# 
# # Create classification forest model
# ctreepcr_mod <- rand_forest(mtry = tune(),
#                          min_n = tune(),
#                          trees = 1000) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create classification forest workflow
# ctreepcr_wf <- workflow() %>%
#   add_recipe(ctreepcr_rec) %>%
#   add_model(ctreepcr_mod)
# 
# # Grid of values to tune over
# ctreepcr_tg <- grid_regular(mtry(range = c(1, 9)),
#                          min_n(),
#                          levels = 15)
# 
# # Split data for cross-validation (CV)
# ctreepcr_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# ctreepcr_cv_results <- ctreepcr_wf %>%
#   tune_grid(resamples = ctreepcr_folds,
#             grid = ctreepcr_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# ctreepcr_best_tune <- ctreepcr_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# ctreepcr_final_wf <- ctreepcr_wf %>%
#   finalize_workflow(ctreepcr_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Predict without a classification cutoff--just the raw probabilities
# ctreepcr_preds <- predict(ctreepcr_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=ctreepcr_preds, file="ctreepcr_preds_15levels.csv", delim = ",")

#################################################################
#################################################################
# PENALIZED LOGISTIC REGRESSION W PCR ###########################
#################################################################
#################################################################
# 
# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# library(tidymodels)
# library(tidyverse)
# library(embed)
# library(lme4)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# plogrpcr_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   # Normalize
#   step_normalize(all_predictors()) %>%
#   # PCR
#   step_pca(all_predictors(), threshold = .94)
# 
# 
# # Prep, Bake, and View Recipe
# plogrpcr_prep <- prep(plogrpcr_rec)
# bake(plogrpcr_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create penalized logistic regression model
# plogrpcr_mod <- logistic_reg(mixture = tune(),
#                           penalty = tune()) %>%
#   set_engine("glmnet")
# 
# # Create logistic regression workflow
# plogrpcr_wf <- workflow() %>%
#   add_recipe(plogrpcr_rec) %>%
#   add_model(plogrpcr_mod)
# 
# # Grid of values to tune over
# plogrpcr_tg <- grid_regular(penalty(),
#                          mixture(),
#                          levels = 5)
# 
# # Split data for cross-validation (CV)
# plogrpcr_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# plogrpcr_cv_results <- plogrpcr_wf %>%
#   tune_grid(resamples = plogrpcr_folds,
#             grid = plogrpcr_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# plogrpcr_best_tune <- plogrpcr_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# plogrpcr_final_wf <- plogrpcr_wf %>%
#   finalize_workflow(plogrpcr_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Predict without a classification cutoff--just the raw probabilities
# plogrpcr_preds_no_c <- predict(plogrpcr_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=plogrpcr_preds_no_c, file="plogrpcr_preds_no_c_no_step_other.csv", delim = ",")

#################################################################
#################################################################
# MODEL STACKING:                     ###########################
# PENALIZED LOGISTIC REGRESSION W PCR ###########################
# CLASSIFICATION FOREST W 5 LEVELS    ###########################
#################################################################
#################################################################

# # DATA CLEANING -------------------------------------------------
# 
# # Load Libraries
# library(vroom)
# library(tidymodels)
# library(tidyverse)
# library(embed)
# library(lme4)
# library(stacks)
# 
# # Load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# stack1_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   # Normalize
#   step_normalize(all_predictors()) %>%
#   # PCR
#   step_pca(all_predictors(), threshold = .94)
# 
# # Prep, Bake, and View Recipe
# stack1_prep <- prep(stack1_rec)
# bake(stack1_prep, employee_train) %>%
#   slice(1:10)
# 
# # CROSS VALIDATION -------------------------------------------------
# stack1_folds <- vfold_cv(employee_train, 
#                   v = 5, 
#                   repeats = 1) # Split data for CV
# 
# stack1_untuned_model <- control_stack_grid() # Control grid for tuning over a grid
# stack1_tuned_model <- control_stack_resamples() # Control grid for models we aren't tuning
# 
# # PENALIZED LOGISTIC REGRESSION MODELING ------------------------------------
# 
# # Create penalized logistic regression model
# plogrpcr_mod <- logistic_reg(mixture = tune(),
#                           penalty = tune()) %>%
#   set_engine("glmnet")
# 
# # Create logistic regression workflow
# plogrpcr_wf <- workflow() %>%
#   add_recipe(stack1_rec) %>%
#   add_model(plogrpcr_mod)
# 
# # Grid of values to tune over
# plogrpcr_tg <- grid_regular(penalty(),
#                          mixture(),
#                          levels = 5)
# 
# # Tune model
# plogrpcr_fit <- plogrpcr_wf %>%
#   tune_grid(resamples = stack1_folds,
#             grid = plogrpcr_tg,
#             metrics = metric_set(roc_auc),
#             control = stack1_untuned_model)
# 
# # CLASSIFICATION FOREST MODELING ------------------------------------
# 
# # Create classification forest model
# ctree_mod <- rand_forest(mtry = tune(),
#                          min_n = tune(),
#                          trees = 750) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create classification forest workflow
# ctree_wf <- workflow() %>%
#   add_recipe(stack1_rec) %>%
#   add_model(ctree_mod)
# 
# # Grid of values to tune over
# ctree_tg <- grid_regular(mtry(range = c(1, 9)),
#                          min_n(),
#                          levels = 5)
# 
# # Run cross-validation
# ctree_fit <- ctree_wf %>%
#   tune_grid(resamples = stack1_folds,
#             grid = ctree_tg,
#             metrics = metric_set(roc_auc),
#             control = stack1_untuned_model)
# 
# # STACKED MODEL ----------------------------------------------
# 
# # Specify models to include
# stack1_stack <- stacks() %>%
#   add_candidates(plogrpcr_fit) %>%
#   add_candidates(ctree_fit)
# 
# # Fit model w/ LASSO penalized regression meta-learner
# stacked_model1 <- stack1_stack %>%
#   blend_predictions() %>%
#   fit_members()
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Predict without a classification cutoff--just the raw probabilities
# stacked1_preds <- predict(stacked_model1,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=stacked1_preds, file="stacked1_preds.csv", delim = ",")

#################################################################
#################################################################
# Support Vector Machine - Radial              ##################
#################################################################
#################################################################

# DATA CLEANING -------------------------------------------------

# # Load Libraries
# library(vroom)
# library(tidymodels)
# library(tidyverse)
# library(embed)
# library(kernlab)
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# svms_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than .1% of the time into an "other" category
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   # Normalize Numeric Predictors
#   step_normalize(all_numeric_predictors())
# 
# # Prep, Bake, and View Recipe
# svms_prep <- prep(svms_rec)
# bake(svms_prep, employee_train) %>%
#   slice(1:10)
# 
# # MODELING ------------------------------------------------------
# 
# # Create SVMS model
# svms_radial_mod <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# 
# # Create SVM workflow
# svms_radial_wf <- workflow() %>%
#   add_recipe(svms_rec) %>%
#   add_model(svms_radial_mod)
# 
# # Grid of values to tune over
# svms_radial_tg <- grid_regular(rbf_sigma(),
#                                cost(),
#                                levels = 5)
# 
# # Split data for cross-validation (CV)
# svms_radial_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# svms_radial_cv_results <- svms_radial_wf %>%
#   tune_grid(resamples = svms_radial_folds,
#             grid = svms_radial_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# svms_radial_best_tune <- svms_radial_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# svms_radial_final_wf <- svms_radial_wf %>%
#   finalize_workflow(svms_radial_best_tune) %>%
#   fit(data = employee_train)
# 
# # PREDICTIONS ---------------------------------------------------
# 
# # Make predictions
# svms_radial_preds <- predict(svms_radial_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=svms_radial_preds, file="svms_radial_preds.csv", delim = ",")

#################################################################
#################################################################
# BALANCING DATA - SMOTE - FOR ALL PREVIOUS MODELS ##############
#################################################################
#################################################################

# # Load Libraries
# library(vroom) # Loading data
# library(tidyverse) # General Use
# library(tidymodels) # General Modeling
# library(embed) # plogr modeling
# library(lme4) # plogr modeling
# library(naivebayes) # Naive Bayes modeling
# library(discrim) # Naive Bayes modeling
# library(kknn) # K nearest neighbors
# library(kernlab) # SVMS
# library(themis) # Balancing Data - SMOTE
# 
# # Re-load Data
# employee_train <- vroom("train.csv")
# employee_test <- vroom("test.csv")
# 
# # Change ACTION to factor before the recipe, as it isn't included in the test data set
# employee_train$ACTION <- as.factor(employee_train$ACTION)
# 
# # Create Recipe
# smote_rec <- recipe(ACTION ~ ., data = employee_train) %>%
#   # Vroom loads in data w numbers as numeric; turn all of these features into factors
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   # Combine categories that occur less than .01% of the time into an "other" category
#   # step_other(all_nominal_predictors(), threshold = .0001) %>%
#   # Target encoding for all nominal predictors
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   # Normalize Numeric Predictors
#   step_normalize(all_numeric_predictors()) %>%
#   # Use SMOTE to balance data with KNN for 5 nearest neighbors
#   # Alternatively, could have used step_upsample() or step_downsample
#   step_smote(all_outcomes(), neighbors = 5)
# 
# # Prep, Bake, and View Recipe
# smote_prep <- prep(smote_rec)
# #   slice(1:10)

# # LOGISTIC REGRESSION ------------------------------------------------------
# 
# # Create logistic regression model
# logr_mod <- logistic_reg() %>%
#   set_engine("glm")
# 
# # Create and fit logistic regression workflow
# logr_wf <- workflow() %>%
#   add_recipe(smote_rec) %>%
#   add_model(logr_mod) %>%
#   fit(data = employee_train)
# 
# # Predict without a classification cutoff--just the raw probabilities
# logr_preds <- predict(logr_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=logr_preds, file="logr_preds_smote.csv", delim = ",")
# 
# # PENALIZED LOGISTIC REGRESSION --------------------------------------------------
# 
# # Create penalized logistic regression model
# plogr_mod <- logistic_reg(mixture = tune(),
#                           penalty = tune()) %>%
#   set_engine("glmnet")
# 
# # Create logistic regression workflow
# plogr_wf <- workflow() %>%
#   add_recipe(smote_rec) %>%
#   add_model(plogr_mod)
# 
# # Grid of values to tune over
# plogr_tg <- grid_regular(penalty(),
#                          mixture(),
#                          levels = 5)
# 
# # Split data for cross-validation (CV)
# plogr_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# plogr_cv_results <- plogr_wf %>%
#   tune_grid(resamples = plogr_folds,
#             grid = plogr_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# plogr_best_tune <- plogr_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# plogr_final_wf <- plogr_wf %>%
#   finalize_workflow(plogr_best_tune) %>%
#   fit(data = employee_train)
# 
# # Predict without a classification cutoff--just the raw probabilities
# plogr_preds <- predict(plogr_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=plogr_preds, file="plogr_preds_smote.csv", delim = ",")
# 
# 
# # RANDOM FOREST ------------------------------------------------------
# 
# # Create classification forest model
# ctree_mod <- rand_forest(mtry = tune(),
#                          min_n = tune(),
#                          trees = 750) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# 
# # Create classification forest workflow
# ctree_wf <- workflow() %>%
#   add_recipe(smote_rec) %>%
#   add_model(ctree_mod)
# 
# # Grid of values to tune over
# ctree_tg <- grid_regular(mtry(range = c(1, 9)),
#                          min_n(),
#                          levels = 5)
# 
# # Split data for cross-validation (CV)
# ctree_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# ctree_cv_results <- ctree_wf %>%
#   tune_grid(resamples = ctree_folds,
#             grid = ctree_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# ctree_best_tune <- ctree_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# ctree_final_wf <- ctree_wf %>%
#   finalize_workflow(ctree_best_tune) %>%
#   fit(data = employee_train)
# 
# # Predict without a classification cutoff--just the raw probabilities
# ctree_preds <- predict(ctree_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=ctree_preds, file="ctree_preds_smote.csv", delim = ",")

# # SVMS Linear ------------------------------------------------------
# 
# # Create SVMS model
# svms_linear_mod <- svm_linear(cost = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# # Create SVM workflow
# svms_linear_wf <- workflow() %>%
#   add_recipe(smote_rec) %>%
#   add_model(svms_linear_mod)
# 
# # Grid of values to tune over
# svms_linear_tg <- grid_regular(cost(),
#                                levels = 5)
# 
# # Split data for cross-validation (CV)
# svms_linear_folds <- vfold_cv(employee_train, v = 5, repeats = 1)
# 
# # Run cross-validation
# svms_linear_cv_results <- svms_linear_wf %>%
#   tune_grid(resamples = svms_linear_folds,
#             grid = svms_linear_tg,
#             metrics = metric_set(roc_auc))
# 
# # Find best tuning parameters
# svms_linear_best_tune <- svms_linear_cv_results %>%
#   select_best("roc_auc")
# 
# # Finalize workflow and fit it
# svms_linear_final_wf <- svms_linear_wf %>%
#   finalize_workflow(svms_linear_best_tune) %>%
#   fit(data = employee_train)
# 
# # Make predictions
# svms_linear_preds <- predict(svms_linear_final_wf,
#                      new_data = employee_test,
#                      type = "prob") %>%
#   bind_cols(employee_test$id, .) %>%
#   rename(Id = ...1) %>%
#   rename(Action = .pred_1) %>%
#   select(Id, Action)
# 
# # Create a CSV with the predictions
# vroom_write(x=svms_linear_preds, file="svms_linear_preds_smote.csv", delim = ",")

########################################################################
########################################################################
#############################
# End run in parallel
# stopCluster(cl)
#############################
########################################################################
########################################################################







#################################################################
#################################################################
# MEGA FOREST                                      ##############
#################################################################
#################################################################

# Load Libraries
library(vroom) # Loading data
library(tidyverse) # General Use
library(tidymodels) # General Modeling
library(embed) # plogr modeling
library(lme4) # plogr modeling
library(naivebayes) # Naive Bayes modeling
library(discrim) # Naive Bayes modeling
library(kknn) # K nearest neighbors
library(kernlab) # SVMS
library(themis) # Balancing Data - SMOTE
library(discrim) # PCA

# Re-load Data
employee_train <- vroom("train.csv")
employee_test <- vroom("test.csv")

# Change ACTION to factor before the recipe, as it isn't included in the test data set
employee_train$ACTION <- as.factor(employee_train$ACTION)

# Create Recipe
rf_rec <- recipe(ACTION ~ ., data = employee_train) %>%
  # Vroom loads in data w numbers as numeric; turn all of these features into factors
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  # Target encoding for all nominal predictors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  # Normalize Numeric Predictors
  step_normalize(all_numeric_predictors()) %>%
  # Use SMOTE to balance data with KNN for 5 nearest neighbors
  step_smote(all_outcomes(), neighbors = 5)

# Prep, Bake, and View Recipe
rf_prep <- prep(rf_rec)

# MODELING ------------------------------------------------------

# Create random forest model
rf_mod <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Create classification forest workflow
rf_wf <- workflow() %>%
  add_recipe(rf_rec) %>%
  add_model(rf_mod)

# Grid of values to tune over
rf_tg <- grid_regular(mtry(range = c(1, 9)),
                         min_n(),
                         levels = 20)

# Split data for cross-validation (CV)
rf_folds <- vfold_cv(employee_train, v = 5, repeats = 1)

# Run cross-validation
rf_cv_results <- rf_wf %>%
  tune_grid(resamples = rf_folds,
            grid = rf_tg,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
rf_best_tune <- rf_cv_results %>%
  select_best("roc_auc")

# Finalize workflow and fit it
rf_final_wf <- rf_wf %>%
  finalize_workflow(rf_best_tune) %>%
  fit(data = employee_train)

# PREDICTIONS ---------------------------------------------------

# Predict without a classification cutoff--just the raw probabilities
rf_preds <- predict(rf_final_wf,
                     new_data = employee_test,
                     type = "prob") %>%
  bind_cols(employee_test$id, .) %>%
  rename(Id = ...1) %>%
  rename(Action = .pred_1) %>%
  select(Id, Action)

# Create a CSV with the predictions
vroom_write(x=rf_preds, file="rf_preds_20levels_SMOTE.csv", delim = ",")

