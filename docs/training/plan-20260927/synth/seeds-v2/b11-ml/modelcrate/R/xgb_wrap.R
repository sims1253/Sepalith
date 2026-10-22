#' Encode an outcome for xgboost
#'
#' Converts the outcome to the numeric coding xgboost expects and picks a
#' matching objective and evaluation metric.
#'
#' @param y The outcome vector.
#' @noRd
xgb_encode_outcome <- function(y) {
  if (is.numeric(y)) {
    return(list(label = y, objective = "reg:squarederror", eval_metric = "rmse",
      classes = NULL))
  }

  # xgboost wants zero-based class codes
  y <- factor(y)
  classes <- levels(y)
  label <- as.integer(y) - 1L
  if (length(classes) == 2L) {
    list(label = label, objective = "binary:logistic", eval_metric = "logloss",
      classes = classes)
  } else {
    list(label = label, objective = "multi:softprob", eval_metric = "mlogloss",
      classes = classes)
  }
}

#' Build an xgboost DMatrix
#'
#' One-hot encodes predictors into a sparse matrix aligned to a stored column
#' layout and wraps it, with an optional label, in an `xgb.DMatrix`.
#'
#' @inheritParams glmnet_matrix
#' @param label Optional numeric label vector.
#' @return An `xgb.DMatrix` with attribute `layout`.
#' @noRd
xgb_dmatrix <- function(data, predictors, label = NULL, layout = NULL) {
  form <- stats::reformulate(predictors, intercept = FALSE)
  old <- options(na.action = "na.pass")
  on.exit(options(old), add = TRUE)
  x <- Matrix::sparse.model.matrix(form, data = data)

  if (!is.null(layout)) {
    absent <- setdiff(layout, colnames(x))
    if (length(absent) > 0) {
      filler <- Matrix::Matrix(0, nrow(x), length(absent), sparse = TRUE,
        dimnames = list(NULL, absent))
      x <- cbind(x, filler)
    }
    x <- x[, layout, drop = FALSE]
  }

  dm <- if (is.null(label)) xgboost::xgb.DMatrix(x) else xgboost::xgb.DMatrix(x, label = label)
  attr(dm, "layout") <- colnames(x)
  dm
}

#' Fit a gradient boosted model with early stopping
#'
#' Holds out a validation fraction, trains with early stopping on it, and
#' keeps the best iteration for prediction.
#'
#' @inheritParams fit_glmnet_model
#' @param params Named list of booster parameters, merged over sensible defaults.
#' @param nrounds Maximum number of boosting rounds.
#' @param valid_frac Fraction of rows held out for early stopping.
#' @param seed Integer seed for the validation split.
#' @return An object of class `mc_xgb`.
#' @family model wrappers
#' @export
fit_xgb_model <- function(data, outcome, predictors = setdiff(names(data), outcome),
                          params = list(), nrounds = 2000, valid_frac = 0.2, seed = 1) {
  enc <- xgb_encode_outcome(data[[outcome]])
  n <- nrow(data)
  n_valid <- floor(valid_frac * n)
  if (n_valid < 1 || n_valid >= n) {
    stop("`valid_frac` must leave at least one row in each of train and valid.",
      call. = FALSE)
  }

  # a logical mask is safe where data[-integer(0), ] would drop every row
  set.seed(seed)
  in_valid <- seq_len(n) %in% sample(n, n_valid)
  dtrain <- xgb_dmatrix(data[!in_valid, , drop = FALSE], predictors, enc$label[!in_valid])
  layout <- attr(dtrain, "layout")
  dvalid <- xgb_dmatrix(data[in_valid, , drop = FALSE], predictors, enc$label[in_valid],
    layout)

  defaults <- list(
    objective = enc$objective,
    eval_metric = enc$eval_metric,
    eta = 0.05,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 2
  )
  if (!is.null(enc$classes) && length(enc$classes) > 2L) {
    defaults$num_class <- length(enc$classes)
  }

  booster <- xgboost::xgb.train(
    params = utils::modifyList(defaults, params),
    data = dtrain,
    nrounds = nrounds,
    watchlist = list(train = dtrain, valid = dvalid),
    early_stopping_rounds = 50,
    verbose = 0
  )

  structure(
    list(booster = booster, layout = layout, predictors = predictors,
      outcome = outcome, classes = enc$classes, objective = enc$objective),
    class = c("mc_xgb", "mc_model")
  )
}

#' Predict from a wrapped xgboost model
#'
#' Returns numeric predictions for regression, event probabilities for binary
#' outcomes, and a probability matrix with class names for multiclass.
#'
#' @param object An object of class `mc_xgb`.
#' @param newdata A data frame of predictors.
#' @param ... Ignored.
#' @return A numeric vector or a probability matrix.
#' @export
predict.mc_xgb <- function(object, newdata, ...) {
  dm <- xgb_dmatrix(newdata, object$predictors, layout = object$layout)
  best <- object$booster$best_iteration
  pred <- stats::predict(object$booster, dm, iterationrange = c(1, best + 1))

  # softprob returns one long vector, row-major by observation
  if (object$objective == "multi:softprob") {
    pred <- matrix(pred, ncol = length(object$classes), byrow = TRUE,
      dimnames = list(NULL, object$classes))
  }
  pred
}

#' Gain importance of a wrapped xgboost model
#'
#' Parses `xgb.importance()` into a plain data frame sorted by gain.
#'
#' @param object An object of class `mc_xgb`.
#' @param top Maximum number of features to return.
#' @return A data frame with columns `feature`, `gain`, `cover`, `frequency`.
#' @export
xgb_importance_table <- function(object, top = 20L) {
  imp <- xgboost::xgb.importance(feature_names = object$layout, model = object$booster)
  out <- data.frame(
    feature = imp$Feature,
    gain = imp$Gain,
    cover = imp$Cover,
    frequency = imp$Frequency
  )
  out <- out[order(-out$gain), , drop = FALSE]
  utils::head(out, top)
}

#' Evaluation log of a wrapped xgboost model
#'
#' Reshapes the training and validation metric history into a long data frame
#' suitable for plotting learning curves.
#'
#' @param object An object of class `mc_xgb`.
#' @return A data frame with columns `iter`, `set` and `value`.
#' @export
xgb_eval_history <- function(object) {
  log <- as.data.frame(object$booster$evaluation_log)
  value_cols <- setdiff(names(log), "iter")

  # columns are named like "train_rmse" and "valid_rmse"
  pieces <- lapply(value_cols, function(col) {
    data.frame(iter = log$iter, set = sub("_.*$", "", col), value = log[[col]])
  })
  do.call(rbind, pieces)
}

#' Print a wrapped xgboost model
#'
#' Shows the objective, best iteration and number of features.
#'
#' @param x An object of class `mc_xgb`.
#' @param ... Ignored.
#' @export
print.mc_xgb <- function(x, ...) {
  cat(sprintf("<mc_xgb: %s, %d features>\n", x$objective, length(x$layout)))
  cat(sprintf("  best iteration = %d (score %.4g)\n",
    as.integer(x$booster$best_iteration), x$booster$best_score))
  invisible(x)
}
