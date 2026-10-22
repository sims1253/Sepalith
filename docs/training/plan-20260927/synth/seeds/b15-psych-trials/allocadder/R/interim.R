#' O'Brien-Fleming type alpha spending
#'
#' Lan-DeMets spending function that spends very little alpha at early looks.
#'
#' @param t Information fractions in (0, 1].
#' @param alpha Overall two-sided type I error rate.
#' @return Cumulative alpha spent at each information fraction.
#' @family spending functions
#' @export
spendObrienFleming <- function(t, alpha = 0.05) {
  stopifnot(all(t > 0 & t <= 1))
  2 - 2 * stats::pnorm(stats::qnorm(1 - alpha / 2) / sqrt(t))
}

#' Pocock type alpha spending
#'
#' Lan-DeMets spending function that spends alpha roughly evenly over looks.
#'
#' @param t Information fractions in (0, 1].
#' @param alpha Overall two-sided type I error rate.
#' @return Cumulative alpha spent at each information fraction.
#' @family spending functions
#' @export
spendPocock <- function(t, alpha = 0.05) {
  stopifnot(all(t > 0 & t <= 1))
  alpha * log(1 + (exp(1) - 1) * t)
}

#' Power family alpha spending
#'
#' Kim-DeMets spending function `alpha * t^rho`; `rho = 3` behaves much like
#' O'Brien-Fleming and `rho = 1` much like Pocock.
#'
#' @param t Information fractions in (0, 1].
#' @param alpha Overall two-sided type I error rate.
#' @param rho Shape parameter of the power family.
#' @return Cumulative alpha spent at each information fraction.
#' @family spending functions
#' @export
spendPowerFamily <- function(t, alpha = 0.05, rho = 2) {
  stopifnot(all(t > 0 & t <= 1), rho > 0)
  alpha * t^rho
}

#' Information fractions for equally spaced looks
#'
#' @param nLooks Total number of analyses, including the final one.
#' @return A numeric vector of length `nLooks` ending in 1.
#' @keywords internal
equalLooks <- function(nLooks) {
  stopifnot(nLooks >= 1)
  seq_len(nLooks) / nLooks
}

#' Two-sided group-sequential efficacy boundaries
#'
#' Turns an alpha spending function into z-scale critical values by Monte
#' Carlo simulation of the joint null distribution of the interim statistics.
#'
#' @param t Information fractions in (0, 1], increasing and ending in 1.
#' @param alpha Overall two-sided type I error rate.
#' @param spending Spending function: `"obf"`, `"pocock"` or `"power"`.
#' @param rho Shape parameter used when `spending = "power"`.
#' @param nSim Number of simulated trials under the null.
#' @param seed Random seed for the simulation.
#' @return A data frame with one row per look: `look`, `t`, `alphaCum`,
#'   `alphaInc`, `zBound` and `pNominal`.
#' @export
spendingBoundaries <- function(t = equalLooks(3), alpha = 0.05,
                               spending = c("obf", "pocock", "power"),
                               rho = 2, nSim = 1e5, seed = 2024) {
  spending <- match.arg(spending)
  if (is.unsorted(t, strictly = TRUE) || utils::tail(t, 1) != 1) {
    stop("`t` must be strictly increasing and end at 1.", call. = FALSE)
  }
  alphaCum <- switch(spending,
    obf = spendObrienFleming(t, alpha),
    pocock = spendPocock(t, alpha),
    power = spendPowerFamily(t, alpha, rho)
  )
  alphaInc <- diff(c(0, alphaCum))
  set.seed(seed)
  # Brownian motion at the looks gives the canonical joint distribution
  increments <- matrix(stats::rnorm(nSim * length(t)), nrow = nSim)
  increments <- sweep(increments, 2, sqrt(diff(c(0, t))), `*`)
  z <- abs(base::t(apply(increments, 1, cumsum)) / rep(sqrt(t), each = nSim))
  alive <- rep(TRUE, nSim)
  zBound <- numeric(length(t))
  for (k in seq_along(t)) {
    nCross <- round(alphaInc[k] * nSim)
    zAlive <- sort(z[alive, k], decreasing = TRUE)
    zBound[k] <- if (nCross < 1) Inf else zAlive[nCross]
    alive <- alive & z[, k] < zBound[k]
  }
  data.frame(
    look = seq_along(t),
    t = t,
    alphaCum = alphaCum,
    alphaInc = alphaInc,
    zBound = zBound,
    pNominal = 2 * stats::pnorm(-zBound)
  )
}

#' Haybittle-Peto boundaries
#'
#' A fixed, very strict interim boundary with the final look tested close to
#' the nominal level.
#'
#' @param nLooks Total number of analyses, including the final one.
#' @param alpha Overall two-sided type I error rate.
#' @param zInterim Critical value used at every interim look.
#' @return A data frame with one row per look: `look`, `zBound` and `pNominal`.
#' @export
haybittlePeto <- function(nLooks, alpha = 0.05, zInterim = 3) {
  zBound <- c(rep(zInterim, nLooks - 1), stats::qnorm(1 - alpha / 2))
  data.frame(
    look = seq_len(nLooks),
    zBound = zBound,
    pNominal = 2 * stats::pnorm(-zBound)
  )
}

#' Decision at an interim analysis
#'
#' Compares an observed z statistic with the boundary for the current look.
#'
#' @param z Observed z statistic at the current look.
#' @param bounds Boundary table from [spendingBoundaries()] or [haybittlePeto()].
#' @param look Index of the current look.
#' @param futilityZ Optional z value below which the trial stops for futility.
#' @return A character string: `"stop_efficacy"`, `"stop_futility"`,
#'   `"continue"` or `"final_not_significant"`.
#' @export
interimDecision <- function(z, bounds, look, futilityZ = NULL) {
  if (look < 1 || look > nrow(bounds)) {
    stop("`look` must be between 1 and ", nrow(bounds), ".", call. = FALSE)
  }
  isFinal <- look == nrow(bounds)
  if (abs(z) >= bounds$zBound[look]) {
    return("stop_efficacy")
  }
  if (isFinal) {
    return("final_not_significant")
  }
  if (!is.null(futilityZ) && z < futilityZ) {
    return("stop_futility")
  }
  "continue"
}

#' Format a boundary table for a statistical analysis plan
#'
#' Rounds and relabels the columns so the table can be pasted into a report.
#'
#' @param bounds Boundary table from [spendingBoundaries()] or [haybittlePeto()].
#' @param digits Number of decimal places for the z bounds.
#' @return A character data frame.
#' @export
formatBoundaries <- function(bounds, digits = 3) {
  out <- data.frame(Look = bounds$look)
  if (!is.null(bounds$t)) {
    out$Information <- sprintf("%.0f%%", 100 * bounds$t)
  }
  out$`Z boundary` <- formatC(bounds$zBound, format = "f", digits = digits)
  out$`Nominal p` <- format.pval(bounds$pNominal, digits = 2, eps = 1e-4)
  out
}
