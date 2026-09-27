xb_dia <- function(X, y, tune = FALSE, cv_folds = 5, tune_length = 20) {
  if (tune) {
    ctrl <- caret::trainControl(
      method = "cv",
      number = cv_folds,
      classProbs = TRUE,
      summaryFunction = caret::twoClassSummary,
      search = "random",
      allowParallel = TRUE
    )

    grid <- expand.grid(
      nrounds = c(50, 100),
      max_depth = c(2, 3, 4, 6),
      eta = c(0.01, 0.05, 0.1, 0.3),
      gamma = c(0, 0.1, 0.5, 1, 2),
      colsample_bytree = c(0.5, 0.7, 0.9, 1),
      min_child_weight = c(1, 3, 5, 7),
      subsample = c(0.6, 0.8, 1)
    )

    model <- caret::train(
      x = X,
      y = y,
      method = "xgbTree",
      metric = "ROC",
      trControl = ctrl,
      tuneLength = tune_length
    )
  } else {
    ctrl <- caret::trainControl(
      method = "cv",
      number = cv_folds,
      classProbs = TRUE,
      summaryFunction = caret::twoClassSummary
    )

    grid <- expand.grid(
      nrounds = 100,
      max_depth = 3,
      eta = 0.3,
      gamma = 0,
      colsample_bytree = 1,
      min_child_weight = 1,
      subsample = 1
    )

    model <- caret::train(
      x = X,
      y = y,
      method = "xgbTree",
      metric = "ROC",
      trControl = ctrl,
      tuneGrid = grid
    )
  }

  return(model)
}