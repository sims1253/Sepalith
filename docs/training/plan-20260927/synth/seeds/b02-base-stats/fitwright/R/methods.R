#' Format p-values for display
#'
#' @param p Numeric vector of p-values.
#' @param digits Number of significant digits.
#' @return A character vector.
#' @noRd
format_p <- function(p, digits = 3) {
  eps <- 10^-digits
  ifelse(p < eps, paste0("< ", format(eps, scientific = FALSE)), format(signif(p, digits)))
}

#' Print a bootstrap result
#'
#' @param x An object of class `fw_boot`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family fw_boot methods
#' @export
print.fw_boot <- function(x, digits = 4, ...) {
  cat("<fw_boot> ", x$R, " resamples\n", sep = "")
  cat("Call: ", paste(deparse(x$call), collapse = "\n"), "\n\n", sep = "")
  tab <- cbind(
    original = x$t0,
    bias = colMeans(x$t, na.rm = TRUE) - x$t0,
    std_error = apply(x$t, 2, stats::sd, na.rm = TRUE)
  )
  print(signif(tab, digits))
  invisible(x)
}

#' Summarise a bootstrap result
#'
#' @param object An object of class `fw_boot`.
#' @param level Confidence level for the reported intervals.
#' @param ... Ignored.
#' @return An object of class `summary.fw_boot`.
#' @family fw_boot methods
#' @export
summary.fw_boot <- function(object, level = 0.95, ...) {
  ci <- boot_ci_percentile(object, level)
  table <- data.frame(
    term = names(object$t0),
    original = unname(object$t0),
    bias = unname(colMeans(object$t, na.rm = TRUE) - object$t0),
    std_error = unname(apply(object$t, 2, stats::sd, na.rm = TRUE)),
    lower = unname(ci[, "lower"]),
    upper = unname(ci[, "upper"]),
    row.names = NULL
  )
  failed <- sum(!stats::complete.cases(object$t))
  structure(
    list(table = table, level = level, R = object$R, failed = failed),
    class = "summary.fw_boot"
  )
}

#' Print a bootstrap summary
#'
#' @param x An object of class `summary.fw_boot`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family fw_boot methods
#' @export
print.summary.fw_boot <- function(x, digits = 4, ...) {
  cat("Bootstrap summary (", x$R, " resamples, ", x$level * 100, "% percentile CI)\n",
    sep = ""
  )
  if (x$failed > 0) {
    cat(x$failed, " resample(s) failed and were dropped\n", sep = "")
  }
  tab <- x$table
  num <- vapply(tab, is.numeric, logical(1))
  tab[num] <- lapply(tab[num], signif, digits = digits)
  print(tab, row.names = FALSE)
  invisible(x)
}

#' Print a permutation test result
#'
#' @param x An object of class `fw_perm`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family fw_perm methods
#' @export
print.fw_perm <- function(x, digits = 4, ...) {
  cat(x$method, "\n", sep = "")
  cat("  statistic  = ", format(signif(x$statistic, digits)), "\n", sep = "")
  cat("  p-value    = ", format_p(x$p_value), "\n", sep = "")
  cat("  alternative: ", x$alternative, " (", x$n_perm, " permutations)\n", sep = "")
  invisible(x)
}

#' Summarise a permutation test result
#'
#' @param object An object of class `fw_perm`.
#' @param probs Quantiles of the null distribution to report.
#' @param ... Ignored.
#' @return An object of class `summary.fw_perm`.
#' @family fw_perm methods
#' @export
summary.fw_perm <- function(object, probs = c(0.025, 0.5, 0.975), ...) {
  p <- object$p_value
  B <- object$n_perm
  structure(
    list(
      method = object$method,
      statistic = object$statistic,
      p_value = p,
      mc_error = sqrt(p * (1 - p) / B),
      null_quantiles = stats::quantile(object$null_dist, probs = probs),
      n_perm = B
    ),
    class = "summary.fw_perm"
  )
}

#' Print a permutation test summary
#'
#' @param x An object of class `summary.fw_perm`.
#' @param digits Number of significant digits to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @family fw_perm methods
#' @export
print.summary.fw_perm <- function(x, digits = 4, ...) {
  cat(x$method, "\n", sep = "")
  cat("  statistic  = ", format(signif(x$statistic, digits)), "\n", sep = "")
  cat("  p-value    = ", format_p(x$p_value), " (MC s.e. ",
    format(signif(x$mc_error, 2)), ")\n",
    sep = ""
  )
  cat("Null distribution quantiles:\n")
  print(signif(x$null_quantiles, digits))
  invisible(x)
}
