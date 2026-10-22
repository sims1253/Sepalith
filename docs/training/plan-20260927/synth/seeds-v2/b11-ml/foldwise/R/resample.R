#' Extract the rows of a fold
#'
#' `fold_analysis()` subsets a data frame to the rows used for fitting in one
#' fold; `fold_assessment()` to its held-out rows.
#'
#' @param data A data frame of predictors and outcome.
#' @param fold One element of an `fw_splits` object.
#' @return A data frame.
#' @export
fold_analysis <- function(data, fold) {
  rows <- fold$analysis
  if (any(rows > nrow(data))) {
    stop("Fold refers to rows beyond the end of `data`.", call. = FALSE)
  }
  data[rows, , drop = FALSE]
}

#' @rdname fold_analysis
#' @export
fold_assessment <- function(data, fold) {
  rows <- fold$assessment
  if (any(rows > nrow(data))) {
    stop("Fold refers to rows beyond the end of `data`.", call. = FALSE)
  }
  data[rows, , drop = FALSE]
}

#' Fit and predict on a single fold
#'
#' Calls `fit_fun` on the analysis rows and `predict_fun` on the assessment
#' rows, returning observed and predicted outcomes side by side.
#'
#' @param data A data frame of predictors and outcome.
#' @param fold One element of an `fw_splits` object.
#' @param outcome Name of the outcome column.
#' @param fit_fun A function taking a data frame and returning a fitted model.
#' @param predict_fun A function taking a model and new data, returning predictions.
#' @keywords internal
run_one_fold <- function(data, fold, outcome, fit_fun, predict_fun) {
  train <- fold_analysis(data, fold)
  test <- fold_assessment(data, fold)

  model <- fit_fun(train)
  estimate <- predict_fun(model, test)
  if (length(estimate) != nrow(test)) {
    stop(sprintf("`predict_fun` returned %d values for %d rows in %s.",
      length(estimate), nrow(test), fold$id), call. = FALSE)
  }

  data.frame(
    id = fold$id,
    .row = fold$assessment,
    truth = test[[outcome]],
    estimate = as.numeric(estimate),
    stringsAsFactors = FALSE
  )
}

#' Cross-validate a model over a set of splits
#'
#' Fits a model on every fold and collects out-of-fold predictions. Errors in
#' individual folds are captured as warnings and the fold is skipped.
#'
#' @param data A data frame of predictors and outcome.
#' @param splits An object of class `fw_splits`.
#' @param outcome Name of the outcome column.
#' @param fit_fun A function taking a data frame and returning a fitted model.
#' @param predict_fun A function taking a model and new data, returning predictions.
#' @return A data frame of out-of-fold predictions with class `fw_oof`.
#' @export
cross_validate <- function(data, splits, outcome, fit_fun, predict_fun = stats::predict) {
  stopifnot(inherits(splits, "fw_splits"), outcome %in% names(data))
  results <- vector("list", length(splits))

  for (i in seq_along(splits)) {
    # one bad fold should not throw away the others
    results[[i]] <- tryCatch(
      run_one_fold(data, splits[[i]], outcome, fit_fun, predict_fun),
      error = function(e) {
        warning(sprintf("%s failed: %s", splits[[i]]$id, conditionMessage(e)), call. = FALSE)
        NULL
      }
    )
  }

  oof <- do.call(rbind, results)
  if (is.null(oof)) {
    stop("All folds failed.", call. = FALSE)
  }
  structure(oof, class = c("fw_oof", "data.frame"), method = attr(splits, "method"))
}

#' Score out-of-fold predictions per fold
#'
#' Applies a metric function separately to each fold's predictions.
#'
#' @param oof A data frame of class `fw_oof`.
#' @param metric A function taking `truth` and `estimate`.
#' @return A data frame with columns `id` and `value`.
#' @export
score_folds <- function(oof, metric = rmse_vec) {
  oof |>
    split(oof$id) |>
    vapply(function(d) metric(d$truth, d$estimate), numeric(1)) |>
    (\(v) data.frame(id = names(v), value = unname(v), stringsAsFactors = FALSE))()
}

#' Summarise fold scores
#'
#' Reduces per-fold metric values to a mean, standard error and count.
#'
#' @param scores A data frame with columns `id` and `value`.
#' @return A one-row data frame with columns `mean`, `std_err` and `n`.
#' @export
summarise_scores <- function(scores) {
  # folds whose metric could not be computed are left out
  values <- scores$value[!is.na(scores$value)]
  n <- length(values)
  if (n == 0L) {
    return(data.frame(mean = NA_real_, std_err = NA_real_, n = 0L))
  }

  std_err <- if (n > 1) stats::sd(values) / sqrt(n) else NA_real_
  data.frame(mean = mean(values), std_err = std_err, n = n)
}

#' Reorder out-of-fold predictions by original row
#'
#' Sorts predictions so that row `i` corresponds to row `i` of the training
#' data, filling rows never held out with `NA`.
#'
#' @param oof A data frame of class `fw_oof`.
#' @param n Number of rows in the original data.
#' @return A numeric vector of length `n`.
#' @export
oof_vector <- function(oof, n) {
  out <- rep(NA_real_, n)

  if (anyDuplicated(oof$.row)) {
    # repeated CV: average every prediction a row received
    agg <- tapply(oof$estimate, oof$.row, mean)
    out[as.integer(names(agg))] <- agg
  } else {
    out[oof$.row] <- oof$estimate
  }
  out
}
