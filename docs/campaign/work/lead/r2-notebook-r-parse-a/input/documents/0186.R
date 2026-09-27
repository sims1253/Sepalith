dt_dia <- function(X, y, tune = FALSE, cv_folds = 5) {
  ctrl <- caret::trainControl(
    method = "cv",
    number = cv_folds,
    classProbs = TRUE,
    summaryFunction = caret::twoClassSummary
  )

  n_samples <- nrow(X)

  grid <- if (tune) {
    expand.grid(
      cp = c(0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
    )
  } else {
    expand.grid(cp = 0.01)
  }

  # 设置rpart控制参数以防止过拟合
  rpart_control <- rpart::rpart.control(
    minsplit = max(2, floor(n_samples * 0.05)),
    minbucket = max(1, floor(n_samples * 0.02)),
    maxdepth = if (tune) 30 else 10,
    xval = 0
  )

  model <- tryCatch(
    {
      caret::train(
        x = X,
        y = y,
        method = "rpart",
        metric = "ROC",
        trControl = ctrl,
        tuneGrid = grid,
        control = rpart_control
      )
    },
    error = function(e) {
      warning(paste(
        "Decision tree training failed with error:",
        e$message,
        "\nRetrying with default parameters..."
      ))

      grid_fallback <- expand.grid(cp = 0.01)
      caret::train(
        x = X,
        y = y,
        method = "rpart",
        metric = "ROC",
        trControl = ctrl,
        tuneGrid = grid_fallback,
        control = rpart::rpart.control(
          minsplit = 20,
          minbucket = 7,
          maxdepth = 10
        )
      )
    }
  )

  return(model)
}