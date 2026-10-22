#' Autocorrelation of a chain at several lags
#'
#' @param x Numeric vector of draws.
#' @param maxLag Largest lag to compute.
#' @return A numeric vector of length `maxLag`, named by lag.
#' @export
autocorrAtLags <- function(x, maxLag = 50L) {
  n <- length(x)
  maxLag <- min(maxLag, n - 1L)
  xc <- x - mean(x)
  denom <- sum(xc^2)
  out <- vapply(seq_len(maxLag), function(k) {
    sum(xc[seq_len(n - k)] * xc[(k + 1):n]) / denom
  }, numeric(1))
  names(out) <- seq_len(maxLag)
  return(out)
}

#' Effective sample size of a chain
#'
#' Sums autocorrelations in pairs until the first negative pair sum, in the
#' spirit of the initial positive sequence estimator.
#'
#' @param x Numeric vector of draws.
#' @param maxLag Largest lag to consider.
#' @return A single effective sample size.
#' @export
effectiveSize <- function(x, maxLag = 200L) {
  n <- length(x)
  if (stats::var(x) == 0) {
    return(0)
  }
  rho <- c(1, autocorrAtLags(x, maxLag))
  nPairs <- floor(length(rho) / 2)
  total <- 0
  for (k in seq_len(nPairs)) {
    pairSum <- rho[2 * k - 1] + rho[2 * k]
    if (pairSum < 0) {
      break
    }
    total <- total + pairSum
  }
  tau <- -1 + 2 * total
  return(n / max(tau, 1))
}

#' Discard burn-in and thin a chain
#'
#' @param chain A matrix of draws, one row per iteration.
#' @param burnIn Number of initial iterations to discard.
#' @param thin Keep every `thin`-th iteration after burn-in.
#' @return The reduced matrix of draws.
#' @export
burnAndThin <- function(chain, burnIn = 0L, thin = 1L) {
  chain <- as.matrix(chain)
  if (burnIn >= nrow(chain)) {
    stop("`burnIn` removes the whole chain.", call. = FALSE)
  }
  keep <- seq.int(burnIn + 1L, nrow(chain), by = thin)
  return(chain[keep, , drop = FALSE])
}

#' Gelman-Rubin potential scale reduction factor
#'
#' @param chains A list of numeric vectors, one per chain, of equal length.
#' @return The potential scale reduction factor.
#' @family convergence diagnostics
#' @export
gelmanRubin <- function(chains) {
  m <- length(chains)
  n <- unique(lengths(chains))
  if (m < 2L || length(n) != 1L) {
    stop("`chains` must hold at least two chains of equal length.", call. = FALSE)
  }
  means <- vapply(chains, mean, numeric(1))
  vars <- vapply(chains, stats::var, numeric(1))
  B <- n * stats::var(means)
  W <- mean(vars)
  varHat <- (n - 1) / n * W + B / n
  return(sqrt(varHat / W))
}

#' Geweke convergence z-score
#'
#' Compares the mean of an early window of the chain to that of a late
#' window, with standard errors scaled by the effective sample size.
#'
#' @param x Numeric vector of draws.
#' @param first Fraction of the chain in the early window.
#' @param last Fraction of the chain in the late window.
#' @return A single z-score.
#' @family convergence diagnostics
#' @export
gewekeZ <- function(x, first = 0.1, last = 0.5) {
  n <- length(x)
  a <- x[seq_len(floor(first * n))]
  b <- x[seq.int(n - floor(last * n) + 1L, n)]
  seA <- stats::sd(a) / sqrt(effectiveSize(a))
  seB <- stats::sd(b) / sqrt(effectiveSize(b))
  return((mean(a) - mean(b)) / sqrt(seA^2 + seB^2))
}

#' Summarise the columns of a chain
#'
#' @param chain A matrix of draws, one row per iteration.
#' @param probs Posterior quantiles to report.
#' @return A data frame of class `chainletSummary`.
#' @export
summarizeChain <- function(chain, probs = c(0.025, 0.5, 0.975)) {
  chain <- as.matrix(chain)
  qs <- t(apply(chain, 2, stats::quantile, probs = probs))
  out <- data.frame(
    param = colnames(chain),
    mean = colMeans(chain),
    sd = apply(chain, 2, stats::sd),
    ess = apply(chain, 2, effectiveSize),
    qs,
    row.names = NULL,
    check.names = FALSE
  )
  class(out) <- c("chainletSummary", class(out))
  return(out)
}

#' Print a chain summary
#'
#' @param x An object of class `chainletSummary`.
#' @param digits Number of significant digits to print.
#' @param ... Passed to [print.data.frame()].
#' @return `x`, invisibly.
#' @export
print.chainletSummary <- function(x, digits = 3, ...) {
  cat("Chain summary:", nrow(x), "parameter(s)\n")
  num <- vapply(x, is.numeric, logical(1))
  shown <- as.data.frame(x)
  shown[num] <- lapply(shown[num], signif, digits = digits)
  shown$ess <- round(shown$ess)
  print.data.frame(shown, row.names = FALSE, ...)
  return(invisible(x))
}
