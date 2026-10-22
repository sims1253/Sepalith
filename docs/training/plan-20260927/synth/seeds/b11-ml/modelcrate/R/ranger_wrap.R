#' Check predictors before a forest fit
#'
#' Ensures every predictor exists and converts character columns to factors,
#' recording their levels so new data can be coerced consistently.
#'
#' @param data A data frame of predictors and outcome.
#' @param predictors Character vector of predictor column names.
#' @return A list with the cleaned `data` and a named list of factor `levels`.
#' @keywords internal
forest.prepare <- function(data, predictors) {
  missing.cols <- setdiff(predictors, names(data))
  if (length(missing.cols) > 0) {
    stop("Unknown predictors: ", paste(missing.cols, collapse = ", "), call. = FALSE)
  }
  levels.map <- list()
  for (col in predictors) {
    if (is.character(data[[col]])) {
      data[[col]] <- factor(data[[col]])
    }
    if (is.factor(data[[col]])) {
      levels.map[[col]] <- levels(data[[col]])
    }
  }
  list(data = data, levels = levels.map)
}

#' Coerce new data to the training factor levels
#'
#' Re-levels each factor predictor to match training, mapping unseen levels
#' to `NA` with a warning.
#'
#' @param newdata A data frame of predictors.
#' @param levels.map Named list of training factor levels.
#' @return `newdata` with factor predictors re-levelled.
#' @keywords internal
forest.align <- function(newdata, levels.map) {
  for (col in names(levels.map)) {
    raw <- as.character(newdata[[col]])
    unseen <- !is.na(raw) & !(raw %in% levels.map[[col]])
    if (any(unseen)) {
      warning(sprintf("Column '%s' has %d unseen level(s); set to NA.", col, sum(unseen)),
        call. = FALSE)
    }
    newdata[[col]] <- factor(raw, levels = levels.map[[col]])
  }
  newdata
}

#' Fit a random forest
#'
#' Wraps [ranger::ranger()] with factor handling and stores the training
#' layout for prediction. Classification is used when the outcome is a factor.
#'
#' @param data A data frame of predictors and outcome.
#' @param outcome Name of the outcome column.
#' @param predictors Character vector of predictor column names.
#' @param num.trees Number of trees to grow.
#' @param probability If `TRUE`, grow a probability forest for classification.
#' @param ... Passed to [ranger::ranger()].
#' @return An object of class `mc_ranger`.
#' @family model wrappers
#' @export
fit.ranger.model <- function(data, outcome, predictors = setdiff(names(data), outcome),
                             num.trees = 500, probability = TRUE, ...) {
  prepared <- forest.prepare(data, predictors)
  y <- prepared$data[[outcome]]
  is.class <- is.factor(y) || is.character(y)
  if (is.class) {
    prepared$data[[outcome]] <- factor(y)
  }
  fit <- ranger::ranger(
    dependent.variable.name = outcome,
    data = prepared$data[c(outcome, predictors)],
    num.trees = num.trees,
    probability = is.class && probability,
    importance = "impurity",
    ...
  )
  structure(
    list(
      fit = fit,
      levels = prepared$levels,
      predictors = predictors,
      outcome = outcome,
      mode = if (is.class) "classification" else "regression"
    ),
    class = c("mc_ranger", "mc_model")
  )
}

#' Predict from a wrapped random forest
#'
#' Aligns factor levels and returns predictions. For probability forests the
#' result is a matrix of class probabilities.
#'
#' @param object An object of class `mc_ranger`.
#' @param newdata A data frame of predictors.
#' @param ... Passed to [ranger::predict.ranger()].
#' @return A numeric vector, factor, or probability matrix.
#' @export
predict.mc_ranger <- function(object, newdata, ...) {
  newdata <- forest.align(newdata[object$predictors], object$levels)
  pred <- stats::predict(object$fit, data = newdata, ...)
  pred$predictions
}

#' Variable importance of a wrapped random forest
#'
#' Returns impurity importance as a data frame sorted from most to least
#' important, optionally scaled to sum to one.
#'
#' @param object An object of class `mc_ranger`.
#' @param normalise If `TRUE`, divide by the total importance.
#' @return A data frame with columns `variable` and `importance`.
#' @export
ranger.importance <- function(object, normalise = FALSE) {
  imp <- ranger::importance(object$fit)
  if (normalise && sum(imp) > 0) {
    imp <- imp / sum(imp)
  }
  out <- data.frame(variable = names(imp), importance = unname(imp))
  out <- out[order(-out$importance), , drop = FALSE]
  rownames(out) <- NULL
  out
}

#' Out-of-bag error of a wrapped random forest
#'
#' Labels the stored prediction error with the loss ranger used for the
#' forest type.
#'
#' @param object An object of class `mc_ranger`.
#' @return A named numeric value: Brier score for probability forests,
#'   misclassification rate for classification, and MSE for regression.
#' @export
ranger.oob.error <- function(object) {
  err <- object$fit$prediction.error
  label <- if (object$mode == "regression") {
    "mse"
  } else if (object$fit$treetype == "Probability estimation") {
    "brier"
  } else {
    "misclassification"
  }
  stats::setNames(err, label)
}

#' Print a wrapped random forest
#'
#' Shows the mode, number of trees and out-of-bag error.
#'
#' @param x An object of class `mc_ranger`.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.mc_ranger <- function(x, ...) {
  oob <- ranger.oob.error(x)
  cat(sprintf("<mc_ranger: %s, %d trees, %d predictors>\n",
    x$mode, x$fit$num.trees, length(x$predictors)))
  cat(sprintf("  OOB %s = %.4g\n", names(oob), oob))
  invisible(x)
}
