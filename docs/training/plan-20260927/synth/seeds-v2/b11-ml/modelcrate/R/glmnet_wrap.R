#' Build a design matrix for glmnet
#'
#' Expands factors into treatment-coded dummies and drops the intercept
#' column, recording the column layout so new data can be aligned to it.
#'
#' @param data A data frame of predictors.
#' @param predictors Character vector of predictor column names.
#' @param layout Optional column layout from a previous call, used for new data.
#' @return A numeric matrix with attribute `layout`.
#' @keywords internal
glmnet_matrix <- function(data, predictors, layout = NULL) {
  form <- stats::reformulate(predictors, intercept = TRUE)
  mf <- stats::model.frame(form, data = data, na.action = stats::na.pass)
  x <- stats::model.matrix(form, mf)[, -1, drop = FALSE]

  if (!is.null(layout)) {
    # columns seen in training but absent now are zero-filled
    absent <- setdiff(layout, colnames(x))
    if (length(absent) > 0) {
      x <- cbind(x, matrix(0, nrow(x), length(absent), dimnames = list(NULL, absent)))
    }
    x <- x[, layout, drop = FALSE]
  }

  attr(x, "layout") <- colnames(x)
  x
}

#' Choose a glmnet family from the outcome
#'
#' Returns `family` unchanged unless it is `"auto"`, in which case the family
#' is guessed from the type and values of the outcome.
#'
#' @param y The outcome vector.
#' @param family One of `"auto"`, `"gaussian"`, `"binomial"`, `"multinomial"` or `"poisson"`.
#' @keywords internal
resolve_glmnet_family <- function(y, family = c("auto", "gaussian", "binomial",
                                               "multinomial", "poisson")) {
  family <- match.arg(family)
  if (family != "auto") {
    return(family)
  }

  if (is.factor(y) || is.character(y) || is.logical(y)) {
    n_classes <- length(unique(stats::na.omit(y)))
    return(if (n_classes == 2L) "binomial" else "multinomial")
  }
  # non-negative whole numbers look like counts
  if (all(y >= 0 & y == round(y), na.rm = TRUE)) "poisson" else "gaussian"
}

#' Fit a cross-validated glmnet model
#'
#' Prepares the design matrix, chooses the family, and fits `cv.glmnet`,
#' storing everything needed for later prediction.
#'
#' @param data A data frame of predictors and outcome.
#' @param outcome Name of the outcome column.
#' @param predictors Character vector of predictor column names.
#' @param alpha Elastic-net mixing parameter between 0 (ridge) and 1 (lasso).
#' @param family Passed to [resolve_glmnet_family()].
#' @param ... Passed to [glmnet::cv.glmnet()].
#' @return An object of class `mc_glmnet`.
#' @family model wrappers
#' @export
fit_glmnet_model <- function(data, outcome, predictors = setdiff(names(data), outcome),
                             alpha = 1, family = "auto", ...) {
  # glmnet does not accept missing values in x or y
  complete <- stats::complete.cases(data[c(outcome, predictors)])
  if (!all(complete)) {
    warning(sprintf("Dropping %d incomplete rows.", sum(!complete)), call. = FALSE)
    data <- data[complete, , drop = FALSE]
  }

  y <- data[[outcome]]
  family <- resolve_glmnet_family(y, family)
  x <- glmnet_matrix(data, predictors)
  cv_fit <- glmnet::cv.glmnet(x, y, alpha = alpha, family = family, ...)

  structure(
    list(
      fit = cv_fit,
      layout = attr(x, "layout"),
      predictors = predictors,
      outcome = outcome,
      family = family,
      alpha = alpha
    ),
    class = c("mc_glmnet", "mc_model")
  )
}

#' Predict from a wrapped glmnet model
#'
#' Aligns new data to the training layout and returns predictions at the
#' chosen penalty.
#'
#' @param object An object of class `mc_glmnet`.
#' @param newdata A data frame of predictors.
#' @param s Penalty to use: `"lambda.min"` or `"lambda.1se"`.
#' @param type Prediction type passed to glmnet, e.g. `"response"` or `"class"`.
#' @param ... Ignored.
#' @return A numeric vector, or a matrix for multinomial probabilities.
#' @export
predict.mc_glmnet <- function(object, newdata, s = c("lambda.min", "lambda.1se"),
                              type = "response", ...) {
  s <- match.arg(s)
  x <- glmnet_matrix(newdata, object$predictors, layout = object$layout)
  pred <- stats::predict(object$fit, newx = x, s = s, type = type)

  if (object$family == "multinomial" && type == "response") {
    # rows x classes x 1; rebuild the matrix so a single row stays a matrix
    dims <- dim(pred)
    return(matrix(pred, nrow = dims[1], ncol = dims[2], dimnames = dimnames(pred)[1:2]))
  }
  drop(pred)
}

#' Tidy coefficients of a wrapped glmnet model
#'
#' Returns the non-zero coefficients at the chosen penalty as a data frame,
#' sorted by absolute size.
#'
#' @param object An object of class `mc_glmnet`.
#' @param s Penalty to use: `"lambda.min"` or `"lambda.1se"`.
#' @return A data frame with columns `term` and `estimate`.
#' @export
glmnet_coef_table <- function(object, s = c("lambda.min", "lambda.1se")) {
  s <- match.arg(s)
  if (object$family == "multinomial") {
    stop("Use `coef()` directly for multinomial fits.", call. = FALSE)
  }

  beta <- as.matrix(stats::coef(object$fit, s = s))
  out <- data.frame(term = rownames(beta), estimate = beta[, 1], row.names = NULL)
  out <- out[out$estimate != 0, , drop = FALSE]
  out[order(-abs(out$estimate)), , drop = FALSE]
}

#' Summarise the cross-validation path
#'
#' Extracts the lambda sequence with its mean cross-validated error, standard
#' error and number of non-zero coefficients.
#'
#' @param object An object of class `mc_glmnet`.
#' @return A data frame with one row per lambda.
#' @export
glmnet_path <- function(object) {
  cv <- object$fit
  data.frame(
    lambda = cv$lambda,
    cv_mean = cv$cvm,
    cv_se = cv$cvsd,
    n_nonzero = as.integer(cv$nzero),
    is_min = cv$lambda == cv$lambda.min,
    is_1se = cv$lambda == cv$lambda.1se
  )
}

#' Print a wrapped glmnet model
#'
#' Shows the family, mixing parameter and selected penalties.
#'
#' @param x An object of class `mc_glmnet`.
#' @param ... Ignored.
#' @export
print.mc_glmnet <- function(x, ...) {
  cat(sprintf("<mc_glmnet: %s, alpha = %.2f, %d predictors>\n",
    x$family, x$alpha, length(x$layout)))
  cat(sprintf("  lambda.min = %.4g, lambda.1se = %.4g\n",
    x$fit$lambda.min, x$fit$lambda.1se))
  invisible(x)
}
