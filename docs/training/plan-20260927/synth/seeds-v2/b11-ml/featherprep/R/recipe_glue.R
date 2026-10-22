#' @importFrom magrittr %>%
NULL

#' Build a standard preprocessing recipe
#'
#' Creates a recipe that imputes, pools rare levels, one-hot encodes nominal
#' predictors, drops zero-variance columns and normalises numeric predictors.
#'
#' @param data A data frame of training data.
#' @param outcome Name of the outcome column.
#' @param idCols Optional character vector of identifier columns kept out of modelling.
#' @param otherThreshold Share below which nominal levels are pooled.
#' @return An untrained `recipes::recipe` object.
#' @export
buildBaseRecipe <- function(data, outcome, idCols = character(), otherThreshold = 0.01) {
  form <- stats::reformulate(".", response = outcome)
  rec <- recipes::recipe(form, data = data)
  if (length(idCols) > 0) {
    rec <- rec %>% recipes::update_role(dplyr::all_of(idCols), new_role = "id")
  }

  # imputation has to run before pooling and dummies see the columns
  rec %>%
    recipes::step_impute_median(recipes::all_numeric_predictors()) %>%
    recipes::step_impute_mode(recipes::all_nominal_predictors()) %>%
    recipes::step_other(recipes::all_nominal_predictors(), threshold = otherThreshold) %>%
    recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE) %>%
    recipes::step_zv(recipes::all_predictors()) %>%
    recipes::step_normalize(recipes::all_numeric_predictors())
}

#' Penalised linear model specification
#'
#' Returns a `parsnip` linear or logistic regression spec using the glmnet
#' engine, with tunable penalty and mixture.
#'
#' @param mode Either `"regression"` or `"classification"`.
#' @param tune If `TRUE`, mark hyperparameters for tuning.
#' @return A `parsnip` model specification.
#' @family model specs
#' @export
glmnetSpec <- function(mode = c("regression", "classification"), tune = TRUE) {
  mode <- match.arg(mode)
  penalty <- if (tune) tune::tune() else 0.01
  mixture <- if (tune) tune::tune() else 1

  spec <- switch(mode,
    regression = parsnip::linear_reg(penalty = penalty, mixture = mixture),
    classification = parsnip::logistic_reg(penalty = penalty, mixture = mixture)
  )
  spec %>% parsnip::set_engine("glmnet")
}

#' Random forest model specification
#'
#' Returns a `parsnip` random forest spec using the ranger engine, with
#' tunable `mtry` and `min_n` and permutation importance enabled.
#'
#' @param mode Either `"regression"` or `"classification"`.
#' @param tune If `TRUE`, mark hyperparameters for tuning.
#' @param trees Number of trees to grow.
#' @return A `parsnip` model specification.
#' @family model specs
#' @export
rangerSpec <- function(mode = c("regression", "classification"), tune = TRUE, trees = 500) {
  mode <- match.arg(mode)
  # NULL lets ranger pick its default mtry from the predictor count
  mtry <- if (tune) tune::tune() else NULL
  minN <- if (tune) tune::tune() else 5

  parsnip::rand_forest(mtry = mtry, min_n = minN, trees = trees) %>%
    parsnip::set_engine("ranger", importance = "permutation") %>%
    parsnip::set_mode(mode)
}

#' Gradient boosted tree model specification
#'
#' Returns a `parsnip` boosted tree spec using the xgboost engine, with
#' tunable depth, learning rate and loss reduction.
#'
#' @param mode Either `"regression"` or `"classification"`.
#' @param tune If `TRUE`, mark hyperparameters for tuning.
#' @param trees Number of boosting rounds.
#' @return A `parsnip` model specification.
#' @family model specs
#' @export
xgboostSpec <- function(mode = c("regression", "classification"), tune = TRUE, trees = 1000) {
  mode <- match.arg(mode)
  treeDepth <- if (tune) tune::tune() else 6
  learnRate <- if (tune) tune::tune() else 0.05
  lossReduction <- if (tune) tune::tune() else 0

  # early stopping needs a validation share, set on the engine below
  parsnip::boost_tree(
    trees = trees,
    tree_depth = treeDepth,
    learn_rate = learnRate,
    loss_reduction = lossReduction,
    stop_iter = 50
  ) %>%
    parsnip::set_engine("xgboost", validation = 0.1) %>%
    parsnip::set_mode(mode)
}

#' Combine a recipe and a model into a workflow
#'
#' Wraps a preprocessing recipe and a model specification in a single
#' workflow object that can be tuned or fitted.
#'
#' @param recipe A `recipes::recipe` object.
#' @param spec A `parsnip` model specification.
#' @return A `workflows::workflow` object.
#' @export
buildWorkflow <- function(recipe, spec) {
  workflows::workflow() %>%
    workflows::add_recipe(recipe) %>%
    workflows::add_model(spec)
}

#' Make a space-filling tuning grid
#'
#' Extracts the tunable parameters of a workflow, finalises any that depend on
#' the data (such as `mtry`), and draws a Latin hypercube grid.
#'
#' @param wf A `workflows::workflow` object.
#' @param data A data frame of training data.
#' @param size Number of candidate parameter sets.
#' @return A tibble of candidate parameter values.
#' @export
makeTuneGrid <- function(wf, data, size = 20) {
  params <- wf %>%
    hardhat::extract_parameter_set_dials()
  if (nrow(params) == 0) {
    stop("Workflow has no tunable parameters.", call. = FALSE)
  }

  # mtry's upper bound depends on the number of predictors after baking
  predictors <- wf %>%
    hardhat::extract_preprocessor() %>%
    recipes::prep(training = data) %>%
    recipes::bake(new_data = NULL, recipes::all_predictors())

  params %>%
    dials::finalize(predictors) %>%
    dials::grid_space_filling(size = size)
}

#' Tune a workflow over resamples
#'
#' Runs a grid search, choosing a default metric set for the workflow's mode.
#'
#' @param wf A `workflows::workflow` object.
#' @param resamples An `rsample` resampling object.
#' @param grid A tibble of candidate parameter values.
#' @param mode Either `"regression"` or `"classification"`.
#' @return A `tune_results` object.
#' @export
tuneWorkflow <- function(wf, resamples, grid, mode = c("regression", "classification")) {
  mode <- match.arg(mode)
  metrics <- switch(mode,
    regression = yardstick::metric_set(yardstick::rmse, yardstick::mae, yardstick::rsq),
    classification = yardstick::metric_set(yardstick::roc_auc, yardstick::mn_log_loss)
  )

  ctrl <- tune::control_grid(save_pred = TRUE, verbose = FALSE, parallel_over = "everything")
  tune::tune_grid(wf, resamples = resamples, grid = grid, metrics = metrics, control = ctrl)
}

#' Finalise a workflow with its best parameters
#'
#' Picks the best candidate by `metric`, optionally applying the one-standard-
#' error rule, and fits the finalised workflow on the full training set.
#'
#' @param wf A `workflows::workflow` object.
#' @param results A `tune_results` object.
#' @param data A data frame of training data.
#' @param metric Name of the metric used for selection.
#' @param oneSe If `TRUE`, prefer the largest penalty within one standard error
#'   (glmnet workflows only).
#' @return A fitted `workflows::workflow` object.
#' @export
finalizeAndFit <- function(wf, results, data, metric = "rmse", oneSe = FALSE) {
  best <- if (oneSe) {
    # larger penalty means a simpler model, so sort it first
    tune::select_by_one_std_err(results, metric = metric, dplyr::desc(penalty))
  } else {
    tune::select_best(results, metric = metric)
  }

  wf %>%
    tune::finalize_workflow(best) %>%
    parsnip::fit(data = data)
}
