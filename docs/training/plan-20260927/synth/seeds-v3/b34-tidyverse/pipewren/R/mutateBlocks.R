#' Rescale a numeric vector
#'
#' @param x A numeric vector.
#' @param method One of `"z"` (centre and divide by the standard deviation),
#'   `"minmax"` (map to `[0, 1]`), `"robust"` (centre on the median and
#'   divide by the IQR), `"unit"` (divide by the Euclidean norm) or `"log"`
#'   (`log1p` of non-negative values).
#' @return A numeric vector the same length as `x`.
#' @family mutate blocks
#' @export
scaleColumn <- function(x, method = c("z", "minmax", "robust", "unit", "log")) {
  method <- match.arg(method)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }

  # Constant input is only centred, never divided by a zero spread
  switch(method,
    z = {
      spread <- stats::sd(x, na.rm = TRUE)
      if (is.na(spread) || spread == 0) return(x - mean(x, na.rm = TRUE))
      (x - mean(x, na.rm = TRUE)) / spread
    },
    minmax = {
      spread <- diff(range(x, na.rm = TRUE))
      if (spread == 0) return(x - min(x, na.rm = TRUE))
      (x - min(x, na.rm = TRUE)) / spread
    },
    robust = {
      spread <- stats::IQR(x, na.rm = TRUE)
      if (spread == 0) return(x - stats::median(x, na.rm = TRUE))
      (x - stats::median(x, na.rm = TRUE)) / spread
    },
    unit = {
      spread <- sqrt(sum(x^2, na.rm = TRUE))
      if (spread == 0) return(x)
      x / spread
    },
    log = {
      if (any(x < 0, na.rm = TRUE)) {
        stop("`x` must be non-negative for `method = \"log\"`.", call. = FALSE)
      }
      log1p(x)
    }
  )
}

#' Moving average with a chosen alignment
#'
#' @param x A numeric vector.
#' @param k Window width.
#' @param align Where the current value sits in the window: `"right"` uses
#'   the current and `k - 1` previous values, `"left"` the current and
#'   `k - 1` following values, and `"center"` an odd window of `k` values
#'   around the current one.
#' @return A numeric vector the same length as `x`, `NA` where the window
#'   is incomplete.
#' @family mutate blocks
#' @export
windowMean <- function(x, k = 3L, align = c("right", "left", "center")) {
  align <- match.arg(align)
  n <- length(x)
  out <- rep(NA_real_, n)
  if (k > n) {
    return(out)
  }

  # Each branch fills the positions that have a complete window
  if (align == "right") {
    for (i in seq(k, n)) {
      window <- x[seq(i - k + 1L, i)]
      out[i] <- mean(window)
    }
  } else if (align == "left") {
    for (i in seq(1L, n - k + 1L)) {
      window <- x[seq(i, i + k - 1L)]
      out[i] <- mean(window)
    }
  } else {
    if (k %% 2L == 0L) {
      stop("`k` must be odd when `align` is \"center\".", call. = FALSE)
    }
    half <- (k - 1L) %/% 2L
    for (i in seq(1L + half, n - half)) {
      window <- x[seq(i - half, i + half)]
      out[i] <- mean(window)
    }
  }
  out
}

#' Flag outlying values
#'
#' @param x A numeric vector.
#' @param method One of `"iqr"` (outside the Tukey fences), `"z"` (absolute
#'   z-score above `cutoff`), `"mad"` (absolute robust z-score above
#'   `cutoff`) or `"quantile"` (outside the `probs` quantiles).
#' @param cutoff Multiplier or score threshold for the `"iqr"`, `"z"` and
#'   `"mad"` methods.
#' @param probs Lower and upper probabilities for the `"quantile"` method.
#' @return A logical vector the same length as `x`, `NA` where `x` is
#'   missing. The bounds used are stored in the attribute `"bounds"`.
#' @family mutate blocks
#' @export
flagOutliers <- function(x, method = c("iqr", "z", "mad", "quantile"),
                         cutoff = NULL, probs = c(0.01, 0.99)) {
  method <- match.arg(method)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }
  defaults <- c(iqr = 1.5, z = 3, mad = 3.5, quantile = NA)
  if (is.null(cutoff)) {
    cutoff <- defaults[[method]]
  }

  # Lower and upper bound for the chosen method
  bounds <- switch(method,
    iqr = {
      q <- stats::quantile(x, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      spread <- q[2L] - q[1L]
      c(q[1L] - cutoff * spread, q[2L] + cutoff * spread)
    },
    z = {
      centre <- mean(x, na.rm = TRUE)
      spread <- stats::sd(x, na.rm = TRUE)
      c(centre - cutoff * spread, centre + cutoff * spread)
    },
    mad = {
      centre <- stats::median(x, na.rm = TRUE)
      spread <- stats::mad(x, na.rm = TRUE)
      c(centre - cutoff * spread, centre + cutoff * spread)
    },
    quantile = {
      stats::quantile(x, probs, na.rm = TRUE, names = FALSE)
    }
  )
  structure(x < bounds[1L] | x > bounds[2L], bounds = bounds)
}

#' Split dates into calendar parts
#'
#' @param x A `Date` vector.
#' @param prefix Optional prefix for the column names, joined with `_`.
#' @return A data frame with the integer columns `year`, `quarter`,
#'   `month`, `week`, `weekday` and `day` and the logical column `weekend`,
#'   each name preceded by `prefix` when given.
#' @family mutate blocks
#' @export
dateParts <- function(x, prefix = NULL) {
  x <- as.Date(x)

  # Each part is derived from one format code
  year <- as.integer(format(x, "%Y"))
  month <- as.integer(format(x, "%m"))
  quarter <- (month - 1L) %/% 3L + 1L
  week <- as.integer(format(x, "%V"))
  weekday <- as.integer(format(x, "%u"))
  day <- as.integer(format(x, "%d"))
  weekend <- weekday >= 6L

  out <- data.frame(
    year = year,
    quarter = quarter,
    month = month,
    week = week,
    weekday = weekday,
    day = day,
    weekend = weekend
  )
  if (!is.null(prefix)) {
    names(out) <- paste(prefix, names(out), sep = "_")
  }
  out
}

#' Add margin, per-unit and share columns
#'
#' @param data A data frame of sales with columns `units`, `revenue`, `cost`
#'   and `discount`.
#' @return `data` with the columns `margin`, `margin_pct`,
#'   `revenue_per_unit`, `cost_per_unit`, `discount_per_unit`,
#'   `margin_per_unit`, `cost_share`, `discount_share` and `margin_share`
#'   added. Ratios with a zero or missing denominator are `NA`.
#' @family mutate blocks
#' @export
addRatios <- function(data) {
  data |>
    mutate(
      # Margin
      margin = revenue - cost - discount,
      margin_pct = dplyr::if_else(revenue > 0, 100 * margin / revenue, NA_real_),
      # Per-unit values
      revenue_per_unit = dplyr::if_else(units > 0, revenue / units, NA_real_),
      cost_per_unit = dplyr::if_else(units > 0, cost / units, NA_real_),
      discount_per_unit = dplyr::if_else(units > 0, discount / units, NA_real_),
      margin_per_unit = dplyr::if_else(units > 0, margin / units, NA_real_),
      # Shares of revenue
      cost_share = dplyr::if_else(revenue > 0, cost / revenue, NA_real_),
      discount_share = dplyr::if_else(revenue > 0, discount / revenue, NA_real_),
      margin_share = dplyr::if_else(revenue > 0, margin / revenue, NA_real_)
    )
}

#' Add lag, difference and growth columns per store
#'
#' @param data A data frame of sales with columns `store`, `date`, `units`,
#'   `revenue`, `cost` and `discount`.
#' @return `data`, sorted by `store` and `date`, with the columns
#'   `units_lag`, `units_diff`, `units_growth`, `revenue_lag`,
#'   `revenue_diff`, `revenue_growth`, `cost_lag`, `cost_diff`,
#'   `cost_growth`, `discount_lag`, `discount_diff` and `discount_growth`
#'   added within each store.
#' @family mutate blocks
#' @export
addLagFeatures <- function(data) {
  data |>
    dplyr::arrange(store, date) |>
    group_by(store) |>
    mutate(
      # Units
      units_lag = dplyr::lag(units),
      units_diff = units - units_lag,
      units_growth = dplyr::if_else(units_lag > 0, units_diff / units_lag, NA_real_),
      # Revenue
      revenue_lag = dplyr::lag(revenue),
      revenue_diff = revenue - revenue_lag,
      revenue_growth = dplyr::if_else(revenue_lag > 0, revenue_diff / revenue_lag, NA_real_),
      # Cost
      cost_lag = dplyr::lag(cost),
      cost_diff = cost - cost_lag,
      cost_growth = dplyr::if_else(cost_lag > 0, cost_diff / cost_lag, NA_real_),
      # Discount
      discount_lag = dplyr::lag(discount),
      discount_diff = discount - discount_lag,
      discount_growth = dplyr::if_else(discount_lag > 0, discount_diff / discount_lag, NA_real_)
    ) |>
    dplyr::ungroup()
}
