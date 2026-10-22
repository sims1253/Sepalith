#' @title Check arguments shared by the rolling-window functions
#' @description Coerces and validates the input series, window width and alignment so the
#'   compiled kernels can assume well-formed arguments.
#' @param x A numeric vector or univariate `ts` object.
#' @param width Window width in observations.
#' @param align Window alignment, one of `"right"`, `"centre"` or `"left"`.
#' @return A list with elements `x`, `width` and `alignCode`.
#' @keywords internal
checkRollArgs <- function(x, width, align) {
  if (is.ts(x) && NCOL(x) > 1L) {
    stop("`x` must be a univariate series.", call. = FALSE)
  }
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }

  if (!is.numeric(width) || length(width) != 1L || is.na(width) || width < 1) {
    stop("`width` must be a single positive number.", call. = FALSE)
  }
  if (width > length(x)) {
    stop(sprintf("`width` (%d) exceeds the series length (%d).", as.integer(width),
      length(x)), call. = FALSE)
  }

  # The kernels take alignment as 0 = left, 1 = centre, 2 = right
  alignCode <- match(align, c("left", "centre", "right")) - 1L
  list(x = as.double(x), width = as.integer(width), alignCode = alignCode)
}

#' @title Restore time-series attributes on a rolled result
#' @param out Numeric vector returned by a compiled kernel.
#' @param x The original input to the rolling function.
#' @noRd
restoreTs <- function(out, x) {
  if (is.ts(x)) {
    return(stats::ts(out, start = stats::start(x), frequency = stats::frequency(x)))
  }
  names(out) <- names(x)
  out
}

#' @title Rolling mean
#' @description Computes the mean over a moving window of fixed width. Positions where the
#'   window is incomplete are `NA`.
#' @param x A numeric vector or univariate `ts` object.
#' @param width Window width in observations.
#' @param align Window alignment, one of `"right"`, `"centre"` or `"left"`.
#' @param naRm Logical; drop missing values inside each window?
#' @return A numeric vector, or `ts`, the same length as `x`.
#' @export
#' @examples
#' rollMean(c(1, 2, 3, 4, 5), width = 3)
rollMean <- function(x, width, align = c("right", "centre", "left"), naRm = FALSE) {
  align <- match.arg(align)
  args <- checkRollArgs(x, width, align)
  out <- rollMeanCpp(args$x, args$width, args$alignCode, naRm)
  restoreTs(out, x)
}

#' @title Rolling standard deviation
#' @description Computes the sample standard deviation over a moving window of fixed
#'   width. Positions where the window is incomplete are `NA`.
#' @inheritParams rollMean
#' @return A numeric vector, or `ts`, the same length as `x`.
#' @export
rollSd <- function(x, width, align = c("right", "centre", "left"), naRm = FALSE) {
  align <- match.arg(align)
  args <- checkRollArgs(x, width, align)
  if (args$width < 2L) {
    stop("`width` must be at least 2 for a standard deviation.", call. = FALSE)
  }

  out <- rollSdCpp(args$x, args$width, args$alignCode, naRm)
  restoreTs(out, x)
}

#' @title Rolling maximum
#' @description Computes the maximum over a moving window of fixed width. Positions where
#'   the window is incomplete are `NA`.
#' @inheritParams rollMean
#' @return A numeric vector, or `ts`, the same length as `x`.
#' @export
rollMax <- function(x, width, align = c("right", "centre", "left"), naRm = FALSE) {
  align <- match.arg(align)
  args <- checkRollArgs(x, width, align)
  out <- rollMaxCpp(args$x, args$width, args$alignCode, naRm)
  restoreTs(out, x)
}

#' @title Rolling sum
#' @description Computes the sum over a moving window of fixed width. Positions where the
#'   window is incomplete are `NA`.
#' @inheritParams rollMean
#' @return A numeric vector, or `ts`, the same length as `x`.
#' @export
rollSum <- function(x, width, align = c("right", "centre", "left"), naRm = FALSE) {
  align <- match.arg(align)
  args <- checkRollArgs(x, width, align)
  out <- rollSumCpp(args$x, args$width, args$alignCode, naRm)
  restoreTs(out, x)
}

#' @title Apply several rolling statistics to grouped data
#' @description Splits a long data frame by site, orders each group by time and appends
#'   rolling statistics of one value column.
#' @param data A data frame in long format with one row per site and time.
#' @param valueCol Name of the numeric column to summarise.
#' @param siteCol Name of the column identifying each site.
#' @param timeCol Name of the column giving the observation time.
#' @param width Window width in observations.
#' @param stats Character vector of statistics to add; any of `"mean"`,
#'   `"sd"`, `"max"` and `"sum"`.
#' @return `data`, reordered by site and time, with one new column per statistic.
#' @export
rollBySite <- function(data, valueCol, siteCol = "site", timeCol = "time", width = 7L,
                       stats = c("mean", "sd", "max", "sum")) {
  stats <- match.arg(stats, several.ok = TRUE)
  if (!all(c(valueCol, siteCol, timeCol) %in% names(data))) {
    stop("`valueCol`, `siteCol` and `timeCol` must be columns of `data`.", call. = FALSE)
  }

  rollers <- list(mean = rollMean, sd = rollSd, max = rollMax, sum = rollSum)
  groups <- split(data, data[[siteCol]])
  out <- vector("list", length(groups))

  for (i in seq_along(groups)) {
    g <- groups[[i]]
    g <- g[order(g[[timeCol]]), , drop = FALSE]

    for (s in stats) {
      newCol <- paste0(valueCol, "_roll", tools::toTitleCase(s))
      # Sites with too few rows get an all-NA column instead of an error
      g[[newCol]] <- if (nrow(g) >= width) {
        rollers[[s]](g[[valueCol]], width = width, naRm = TRUE)
      } else {
        rep(NA_real_, nrow(g))
      }
    }
    out[[i]] <- g
  }

  do.call(rbind, out) |>
    `rownames<-`(NULL)
}

#' @title Rolling anomaly flags
#' @description Flags observations that deviate from the trailing rolling mean by more than
#'   `k` rolling standard deviations.
#' @inheritParams rollMean
#' @param k Threshold in standard deviations.
#' @return A logical vector the same length as `x`; `NA` where the window is
#'   incomplete.
#' @export
rollAnomaly <- function(x, width, k = 3, naRm = TRUE) {
  centre <- rollMean(x, width, align = "right", naRm = naRm) |> as.numeric()
  spread <- rollSd(x, width, align = "right", naRm = naRm) |> as.numeric()

  # Compare each value with the window that ends just before it, so an
  # extreme value cannot inflate its own baseline
  lagged <- c(NA_real_, utils::head(centre, -1L))
  laggedSd <- c(NA_real_, utils::head(spread, -1L))
  abs(as.numeric(x) - lagged) > k * laggedSd
}
