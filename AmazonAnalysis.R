#################################################################
#################################################################
# Amazon Employee Access Challenge    ###########################
# Ryan Wolff                          ###########################
# 6 October 2023                      ###########################
#################################################################
#################################################################

#################################################################
#################################################################
# LOAD DATA                           ###########################
#################################################################
#################################################################

# Data Location and Description
# https://www.kaggle.com/competitions/amazon-employee-access-challenge/data

# Load Libraries
library(vroom)

# Load Data
employee_train <- vroom("train.csv")
employee_test <- vroom("test.csv")

#################################################################
#################################################################
# EDA                                 ###########################
#################################################################
#################################################################

# Load Libraries
library(DataExplorer)
library(patchwork)
library(tidyverse)
library(inspectdf)
library(ggmosaic)

# Create an EDA dataset and correct Vroom's mistake and make numeric data into factors
fact_employee_train_eda <- employee_train
cols <- c("ACTION",
          "RESOURCE", 
          "MGR_ID", 
          "ROLE_ROLLUP_1", 
          "ROLE_ROLLUP_2", 
          "ROLE_DEPTNAME", 
          "ROLE_TITLE",
          "ROLE_FAMILY_DESC",
          "ROLE_FAMILY",
          "ROLE_CODE")
fact_employee_train_eda[cols] <- lapply(fact_employee_train_eda[cols], 
                                   factor)

# Examine Factor Variables:
  # cnt = # unique variables
  # common = most common level
  # common_pcnt = percentage representing most common level
  # levels = list of the proportions of each level of the variable
factor_exploration_plot <- fact_employee_train_eda %>%
  inspect_cat() %>%
  show_plot()
factor_exploration_plot

fact_employee_train_eda %>%
  inspect_cat()

# Create an EDA dataset making every feature into a factor except ACTION, which remains numeric
num_employee_train_eda <- employee_train
cols <- c("RESOURCE", 
          "MGR_ID", 
          "ROLE_ROLLUP_1", 
          "ROLE_ROLLUP_2", 
          "ROLE_DEPTNAME", 
          "ROLE_TITLE",
          "ROLE_FAMILY_DESC",
          "ROLE_FAMILY",
          "ROLE_CODE")
num_employee_train_eda[cols] <- lapply(num_employee_train_eda[cols], 
                                   factor)

# Identify the top 30 most popular RESOURCEs
resource_data <- num_employee_train_eda %>%
  group_by(RESOURCE) %>%
  summarize(mean = mean(ACTION),
            n = n()) %>%
  slice_max(order_by = n,
            n = 30)
resource_data <- resource_data["RESOURCE"]

# Subset the EDA dataset to the 30 most popular RESOURCEs
num_employee_train_eda <- num_employee_train_eda %>%
  filter(RESOURCE %in% resource_data$RESOURCE)

# Bar chart of the top 30 most popular RESOURCEs' ACTION results
action_resources_barcharts <- ggplot(data = num_employee_train_eda,
       aes(x = ACTION)) +
  geom_bar() +
  ggtitle("ACTION Results for 30 Most Common Products") +
  xlab("ACTION: 0 and 1, Respectively") +
  ylab("Count of Each ACTION Result") +
  theme(plot.title = element_text(hjust = .5)) +
  facet_wrap( ~ RESOURCE)
action_resources_barcharts

# Create a 2-Way Plot of Prominent Plots
twoway_patch <- (factor_exploration_plot) / (action_resources_barcharts)
twoway_patch
  
# Findings:
  # inspect_cat():
    # ACTION is extremely homogenous--94.2% are in value 1, aka the resource was approved
    # ROLE_ROLLUP_1 is extremely homogenous--65.3% are in value 117961
  # show_plot() is a visualization of inspect_cat()
  # geom_bar():
    # 20897 is the only RESOURCE that is not overwhelmingly approved

#################################################################
#################################################################
# LOGISTIC REGRESSION                 ###########################
#################################################################
#################################################################

# DATA CLEANING -------------------------------------------------

# Load Libraries
library(tidymodels)

# Re-load Data
employee_train <- vroom("train.csv")
employee_test <- vroom("test.csv")

# Change ACTION to factor before the recipe, as it isn't included in the test data set
employee_train$ACTION <- as.factor(employee_train$ACTION)

# Create Recipe
logr_rec <- recipe(ACTION ~ ., data = employee_train) %>%
  # Vroom loads in data w numbers as numeric; turn all of these features into factors
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  # Combine categories that occur less than 1% of the time into an "other" category
  step_other(all_nominal_predictors(), threshold = .01) %>%
  # Dummy variable encoding for all nominal predictors
  step_dummy(all_nominal_predictors())


# Prep, Bake, and View Recipe
logr_prep <- prep(logr_rec)
bake(logr_prep, employee_train) %>%
  slice(1:10)

# MODELING ------------------------------------------------------

# Create logistic regression model
logr_mod <- logistic_reg() %>%
  set_engine("glm")

# Create and fit logistic regression workflow
logr_wf <- workflow() %>%
  add_recipe(logr_rec) %>%
  add_model(logr_mod) %>%
  fit(data = employee_train)

# Predict with classification cutoff = .70
logr_preds <- predict(logr_wf,
                     new_data = employee_test,
                     type = "prob") %>%
  mutate(ifelse(.pred_1 > .83, 1, 0)) %>%
  bind_cols(employee_test$id, .) %>%
  rename(Id = ...1) %>%
  rename(Action = names(.)[4]) %>%
  select(Id, Action)

# Create a CSV with the predictions
# vroom_write(x=logr_preds, file="logr_preds.csv", delim = ",")

# Predict without a classification cutoff--just the raw probabilities
logr_preds_no_c <- predict(logr_wf,
                     new_data = employee_test,
                     type = "prob") %>%
  bind_cols(employee_test$id, .) %>%
  rename(Id = ...1) %>%
  rename(Action = .pred_1) %>%
  select(Id, Action)

# Create a CSV with the predictions
vroom_write(x=logr_preds_no_c, file="logr_preds_no_c.csv", delim = ",")

#################################################################
#################################################################
# PENALIZED LOGISTIC REGRESSION       ###########################
#################################################################
#################################################################

# DATA CLEANING -------------------------------------------------

# Load Libraries
library(tidymodels)
library(embed)
library(lme4)

# Re-load Data
employee_train <- vroom("train.csv")
employee_test <- vroom("test.csv")

# Change ACTION to factor before the recipe, as it isn't included in the test data set
employee_train$ACTION <- as.factor(employee_train$ACTION)

# Create Recipe
plogr_rec <- recipe(ACTION ~ ., data = employee_train) %>%
  # Vroom loads in data w numbers as numeric; turn all of these features into factors
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  # Combine categories that occur less than .1% of the time into an "other" category
  step_other(all_nominal_predictors(), threshold = .001) %>%
  # Target encoding for all nominal predictors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))


# Prep, Bake, and View Recipe
plogr_prep <- prep(plogr_rec)
bake(plogr_prep, employee_train) %>%
  slice(1:10)

# MODELING ------------------------------------------------------

# Create penalized logistic regression model
plogr_mod <- logistic_reg(mixture = tune(),
                          penalty = tune()) %>%
  set_engine("glmnet")

# Create logistic regression workflow
plogr_wf <- workflow() %>%
  add_recipe(plogr_rec) %>%
  add_model(plogr_mod)

# Grid of values to tune over
plogr_tg <- grid_regular(penalty(),
                         mixture(),
                         levels = 5)

# Split data for cross-validation (CV)
plogr_folds <- vfold_cv(employee_train, v = 5, repeats = 1)

# Run cross-validation
plogr_cv_results <- plogr_wf %>%
  tune_grid(resamples = plogr_folds,
            grid = plogr_tg,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
plogr_best_tune <- plogr_cv_results %>%
  select_best("roc_auc")

# Finalize workflow and fit it
plogr_final_wf <- plogr_wf %>%
  finalize_workflow(plogr_best_tune) %>%
  fit(data = employee_train)

# Predict without a classification cutoff--just the raw probabilities
plogr_preds_no_c <- predict(plogr_final_wf,
                     new_data = employee_test,
                     type = "prob") %>%
  bind_cols(employee_test$id, .) %>%
  rename(Id = ...1) %>%
  rename(Action = .pred_1) %>%
  select(Id, Action)

# Create a CSV with the predictions
vroom_write(x=plogr_preds_no_c, file="plogr_preds_no_c.csv", delim = ",")
