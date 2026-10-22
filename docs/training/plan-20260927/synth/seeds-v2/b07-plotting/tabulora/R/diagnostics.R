#' @title Extract the pieces needed for diagnostic plots
#' @description Pulls fitted values, raw residuals and standardised residuals out of a
#'   fitted model.
#' @param model A fitted model object supporting [stats::fitted()] and
#'   [stats::residuals()], such as an `lm` or `glm` fit.
#' @keywords internal
diagData <- function(model) {
  fit <- stats::fitted(model)
  res <- stats::residuals(model)

  # Not every model class has rstandard(); fall back to scaling by the SD
  std <- tryCatch(
    stats::rstandard(model),
    error = function(e) res / stats::sd(res, na.rm = TRUE)
  )
  data.frame(fitted = fit, resid = res, std_resid = std)
}

#' @title Residuals against fitted values
#' @description Plots residuals against fitted values with a zero line and a lowess
#'   smoother to reveal curvature.
#' @param model A fitted model object supporting [stats::fitted()] and
#'   [stats::residuals()], such as an `lm` or `glm` fit.
#' @param col Point colour.
#' @param main Plot title.
#' @return The diagnostic data, invisibly.
#' @family diagnostics
#' @export
plotResidFitted <- function(model, col = "#3e5366", main = "Residuals vs fitted") {
  d <- diagData(model)

  graphics::plot(d$fitted, d$resid, pch = 19, cex = 0.6, col = col,
                 xlab = "Fitted values", ylab = "Residuals", main = main)
  graphics::abline(h = 0, lty = 2, col = "grey50")
  # A flat smoother means no systematic curvature left in the residuals
  graphics::lines(stats::lowess(d$fitted, d$resid), col = "#c8553d", lwd = 2)
  invisible(d)
}

#' @title Normal quantile plot of standardised residuals
#' @description Plots standardised residuals against normal quantiles with a reference
#'   line through the quartiles.
#' @inheritParams plotResidFitted
#' @param col Point colour.
#' @param main Plot title.
#' @return The diagnostic data, invisibly.
#' @family diagnostics
#' @export
plotResidQQ <- function(model, col = "#3e5366", main = "Normal Q-Q") {
  d <- diagData(model)

  graphics::plot(stats::qqnorm(d$std_resid, plot.it = FALSE), pch = 19, cex = 0.6,
                 col = col, xlab = "Theoretical quantiles",
                 ylab = "Standardised residuals", main = main)
  stats::qqline(d$std_resid, lty = 2, col = "grey50")
  invisible(d)
}

#' @title Scale-location plot
#' @description Plots the square root of absolute standardised residuals against fitted
#'   values to check for non-constant variance.
#' @inheritParams plotResidFitted
#' @param col Point colour.
#' @param main Plot title.
#' @return The diagnostic data, invisibly.
#' @family diagnostics
#' @export
plotScaleLocation <- function(model, col = "#3e5366", main = "Scale-location") {
  d <- diagData(model)
  sqrtAbs <- sqrt(abs(d$std_resid))
  graphics::plot(d$fitted, sqrtAbs, pch = 19, cex = 0.6, col = col,
                 xlab = "Fitted values", ylab = "sqrt(|Standardised residuals|)",
                 main = main)
  graphics::lines(stats::lowess(d$fitted, sqrtAbs), col = "#c8553d", lwd = 2)
  invisible(d)
}

#' @title Residual histogram
#' @description Draws a histogram of standardised residuals with a standard normal
#'   density for comparison.
#' @inheritParams plotResidFitted
#' @param col Bar fill colour.
#' @param main Plot title.
#' @return The diagnostic data, invisibly.
#' @family diagnostics
#' @export
plotResidHist <- function(model, col = "#a7b7c7", main = "Residual distribution") {
  d <- diagData(model)

  graphics::hist(d$std_resid, breaks = "FD", freq = FALSE, col = col,
                 border = "white", xlab = "Standardised residuals", main = main)
  graphics::curve(stats::dnorm(x), add = TRUE, lwd = 2, col = "#c8553d")
  invisible(d)
}

#' @title Four-panel diagnostic display
#' @description Draws the residual, Q-Q, scale-location and histogram panels on a two by
#'   two grid, restoring the graphical parameters afterwards.
#' @inheritParams plotResidFitted
#' @param title Optional overall title drawn in the outer margin.
#' @return The diagnostic data, invisibly.
#' @family diagnostics
#' @export
plotDiagnostics <- function(model, title = NULL) {
  oma <- if (is.null(title)) c(0, 0, 0, 0) else c(0, 0, 2, 0)
  oldPar <- graphics::par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = oma, las = 1)
  on.exit(graphics::par(oldPar), add = TRUE)

  plotResidFitted(model)
  plotResidQQ(model)
  plotScaleLocation(model)
  d <- plotResidHist(model)

  if (!is.null(title)) {
    graphics::mtext(title, side = 3, outer = TRUE, font = 2, line = 0.5)
  }
  invisible(d)
}

#' @title Write the diagnostic display to a PNG file
#' @description Opens a PNG device, draws [plotDiagnostics()] and closes the device even
#'   if plotting fails.
#' @inheritParams plotResidFitted
#' @param path Output file path.
#' @param width,height Device size in pixels.
#' @param ... Passed to [plotDiagnostics()].
#' @return `path`, invisibly.
#' @family diagnostics
#' @export
saveDiagnostics <- function(model, path, width = 1600, height = 1400, ...) {
  grDevices::png(path, width = width, height = height, res = 200)
  # Close the device even if one of the panels errors
  on.exit(grDevices::dev.off(), add = TRUE)

  plotDiagnostics(model, ...)
  invisible(path)
}
