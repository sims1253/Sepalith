#' Small-sample corrected AIC
#'
#' Computes AICc from the log-likelihood of a fitted model, falling back to
#' plain AIC when the correction is undefined.
#'
#' @param object A fitted model object with a [stats::logLik()] method.
#' @return A single numeric value.
#' @keywords internal
aicc <- function(object) {
  ll <- stats::logLik(object)
  k <- attr(ll, "df")
  n <- attr(ll, "nobs")
  if (is.null(n)) {
    n <- stats::nobs(object)
  }
  aic <- -2 * as.numeric(ll) + 2 * k
  if (n - k - 1 <= 0) {
    warning("AICc is undefined for n <= k + 1; returning AIC.", call. = FALSE)
    return(aic)
  }
  aic + (2 * k * (k + 1)) / (n - k - 1)
}

#' Akaike weights
#'
#' Converts a vector of information criteria into relative model weights that
#' sum to one.
#'
#' @param ic A numeric vector of information criterion values.
#' @return A numeric vector the same length as `ic`.
#' @export
akaike_weights <- function(ic) {
  if (!is.numeric(ic) || length(ic) == 0L) {
    stop("`ic` must be a non-empty numeric vector.", call. = FALSE)
  }
  if (anyNA(ic)) {
    stop("`ic` must not contain missing values.", call. = FALSE)
  }
  delta <- ic - min(ic)
  rel <- exp(-0.5 * delta)
  rel / sum(rel)
}

#' Compare fitted models by an information criterion
#'
#' Fits nothing; it collects log-likelihoods and information criteria from
#' already fitted models and orders them from best to worst.
#'
#' @param ... Two or more fitted model objects, optionally named.
#' @param .names Optional character vector of model labels.
#' @param criterion Which information criterion to rank by.
#' @return A data frame of class `ts_ic_table`, one row per model.
#' @export
ic_table <- function(..., .names = NULL, criterion = c("AIC", "BIC", "AICc")) {
  criterion <- match.arg(criterion)
  models <- list(...)
  if (length(models) < 2L) {
    stop("Supply at least two fitted models.", call. = FALSE)
  }
  if (is.null(.names)) {
    .names <- names(models)
    if (is.null(.names) || any(.names == "")) {
      .names <- paste0("model_", seq_along(models))
    }
  }
  n_models <- length(models)
  ic <- numeric(n_models)
  df <- integer(n_models)
  ll <- numeric(n_models)
  for (i in seq_len(n_models)) {
    fit <- models[[i]]
    loglik <- stats::logLik(fit)
    ll[i] <- as.numeric(loglik)
    df[i] <- as.integer(attr(loglik, "df"))
    ic[i] <- switch(criterion,
      AIC = stats::AIC(fit),
      BIC = stats::BIC(fit),
      AICc = aicc(fit)
    )
  }
  out <- data.frame(
    model = .names,
    df = df,
    logLik = ll,
    ic = ic,
    delta = ic - min(ic),
    weight = akaike_weights(ic),
    stringsAsFactors = FALSE
  )
  out <- out[order(out$ic), , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "criterion") <- criterion
  class(out) <- c("ts_ic_table", "data.frame")
  out
}

#' Likelihood-ratio test for nested models
#'
#' Compares a restricted model against a fuller one fitted to the same
#' observations, using the asymptotic chi-squared reference distribution.
#'
#' @param restricted The smaller, nested fitted model.
#' @param full The larger fitted model.
#' @return An object of class `ts_lr_test`.
#' @export
lr_test <- function(restricted, full) {
  if (!identical(class(restricted), class(full))) {
    warning(
      sprintf(
        "Comparing models of different classes (%s vs %s).",
        class(restricted)[1L],
        class(full)[1L]
      ),
      call. = FALSE
    )
  }
  ll0 <- stats::logLik(restricted)
  ll1 <- stats::logLik(full)
  df_diff <- attr(ll1, "df") - attr(ll0, "df")
  if (df_diff <= 0) {
    stop("`full` must have more parameters than `restricted`.", call. = FALSE)
  }
  n0 <- stats::nobs(restricted)
  n1 <- stats::nobs(full)
  if (n0 != n1) {
    stop(
      sprintf("Models use different numbers of observations (%d vs %d).", n0, n1),
      call. = FALSE
    )
  }
  stat <- 2 * (as.numeric(ll1) - as.numeric(ll0))
  if (stat < 0) {
    warning("Negative LR statistic; are the models really nested?", call. = FALSE)
    stat <- 0
  }
  p_value <- stats::pchisq(stat, df = df_diff, lower.tail = FALSE)
  structure(
    list(statistic = stat, df = df_diff, p.value = p_value, nobs = n1),
    class = "ts_lr_test"
  )
}

#' Sequential likelihood-ratio tests
#'
#' Tests each model in a list against the next one, as in a forward build-up
#' of nested models, and adjusts the p-values with Holm's method.
#'
#' @param models A list of nested fitted models, smallest first.
#' @return A data frame with one row per consecutive pair.
#' @export
lr_sequence <- function(models) {
  if (!is.list(models) || length(models) < 2L) {
    stop("`models` must be a list of at least two fitted models.", call. = FALSE)
  }
  labels <- names(models)
  if (is.null(labels) || any(labels == "")) {
    labels <- paste0("model_", seq_along(models))
  }
  n_steps <- length(models) - 1L
  rows <- vector("list", n_steps)
  for (i in seq_len(n_steps)) {
    test <- lr_test(models[[i]], models[[i + 1L]])
    rows[[i]] <- data.frame(
      from = labels[i],
      to = labels[i + 1L],
      statistic = test$statistic,
      df = test$df,
      p.value = test$p.value,
      stringsAsFactors = FALSE
    )
  }
  out <- do.call(rbind, rows)
  out$p.adjusted <- stats::p.adjust(out$p.value, method = "holm")
  out
}

#' Format a model comparison table
#'
#' Rounds the numeric columns and relabels the criterion column for display.
#'
#' @param x An object of class `ts_ic_table`.
#' @param digits Number of decimal places for numeric columns.
#' @param ... Unused.
#' @return A plain data frame of character columns.
#' @export
format.ts_ic_table <- function(x, digits = 2, ...) {
  out <- x
  class(out) <- "data.frame"
  for (col in c("logLik", "ic", "delta")) {
    out[[col]] <- formatC(out[[col]], format = "f", digits = digits)
  }
  out$weight <- formatC(out$weight, format = "f", digits = 3)
  names(out)[names(out) == "ic"] <- attr(x, "criterion")
  out
}

#' Print a model comparison table
#'
#' Prints a header naming the criterion followed by the formatted table.
#'
#' @param x An object of class `ts_ic_table`.
#' @param digits Number of decimal places for numeric columns.
#' @param ... Passed to [print.data.frame()].
#' @return `x`, invisibly.
#' @export
print.ts_ic_table <- function(x, digits = 2, ...) {
  cat("<model comparison by ", attr(x, "criterion"), ">\n", sep = "")
  print(format(x, digits = digits), row.names = FALSE, ...)
  invisible(x)
}

#' Format a likelihood-ratio test
#'
#' Collapses the test statistic, degrees of freedom and p-value into one line.
#'
#' @param x An object of class `ts_lr_test`.
#' @param digits Number of decimal places for numeric columns.
#' @param ... Unused.
#' @return A single string.
#' @export
format.ts_lr_test <- function(x, digits = 3, ...) {
  sprintf(
    "LR = %s, df = %d, p = %s (n = %d)",
    formatC(x$statistic, format = "f", digits = digits),
    as.integer(x$df),
    format.pval(x$p.value, digits = digits),
    as.integer(x$nobs)
  )
}

#' Print a likelihood-ratio test
#'
#' Prints a header followed by the one-line summary of the test.
#'
#' @param x An object of class `ts_lr_test`.
#' @param digits Number of decimal places for numeric columns.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.ts_lr_test <- function(x, digits = 3, ...) {
  cat("<likelihood-ratio test>\n")
  cat(format(x, digits = digits), "\n", sep = "")
  invisible(x)
}
