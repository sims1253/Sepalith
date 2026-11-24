#' Outlier fences for a numeric vector
#'
#' @param x A numeric vector.
#' @param method How the fences are computed: `"iqr"` (Tukey fences),
#'   `"zscore"` (mean plus or minus `k` standard deviations), `"mad"`
#'   (median plus or minus `k` scaled MADs), `"quantile"` (the `k` and
#'   `1 - k` quantiles) or `"log.iqr"` (Tukey fences on the log scale, for
#'   positive values).
#' @param k Multiplier, or tail probability for `"quantile"`.
#' @return A numeric vector with elements `lower` and `upper`.
#' @export
outlier.fences <- function(x, method = c("iqr", "zscore", "mad", "quantile", "log.iqr"),
                           k = NULL) {
  method <- match.arg(method)
  x <- x[!is.na(x)]
  switch(method,
    iqr = {
      k <- if (is.null(k)) 1.5 else k
      q <- stats::quantile(x, c(0.25, 0.75), names = FALSE)
      c(lower = q[1] - k * diff(q), upper = q[2] + k * diff(q))
    },
    zscore = {
      k <- if (is.null(k)) 3 else k
      centre <- mean(x)
      spread <- stats::sd(x)
      c(lower = centre - k * spread, upper = centre + k * spread)
    },
    mad = {
      k <- if (is.null(k)) 3 else k
      centre <- stats::median(x)
      spread <- stats::mad(x)
      c(lower = centre - k * spread, upper = centre + k * spread)
    },
    quantile = {
      k <- if (is.null(k)) 0.01 else k
      q <- stats::quantile(x, c(k, 1 - k), names = FALSE)
      c(lower = q[1], upper = q[2])
    },
    log.iqr = {
      k <- if (is.null(k)) 1.5 else k
      q <- stats::quantile(log(x[x > 0]), c(0.25, 0.75), names = FALSE)
      exp(c(lower = q[1] - k * diff(q), upper = q[2] + k * diff(q)))
    }
  )
}

#' Flag outliers
#'
#' @inheritParams outlier.fences
#' @return A logical vector the same length as `x`, `TRUE` outside the
#'   fences and `FALSE` for missing values.
#' @export
flag.outliers <- function(x, method = "iqr", k = NULL) {
  fences <- outlier.fences(x, method = method, k = k)
  out <- x < fences[["lower"]] | x > fences[["upper"]]
  out[is.na(out)] <- FALSE
  out
}

#' Winsorize a numeric vector
#'
#' @param x A numeric vector.
#' @param lower,upper Limits; values beyond them are set to the limit.
#'   `NULL` leaves that side untouched.
#' @return `x` with extreme values replaced; the attribute `"clipped"` is a
#'   list with the positions changed at the `lower` and `upper` side.
#' @export
winsorize <- function(x, lower = NULL, upper = NULL) {
  if (!is.null(lower) && !is.null(upper) && lower > upper) {
    stop("`lower` must not be greater than `upper`.", call. = FALSE)
  }
  low <- integer()
  high <- integer()
  if (!is.null(lower)) {
    low <- which(x < lower)
    x[low] <- lower
  }
  if (!is.null(upper)) {
    high <- which(x > upper)
    x[high] <- upper
  }
  attr(x, "clipped") <- list(lower = low, upper = high)
  x
}

#' Treat outliers in one column
#'
#' @inheritParams outlier.fences
#' @param action What to do with outliers: `"flag"` leaves `x` unchanged,
#'   `"na"` sets them to `NA`, `"clip"` winsorizes them to the fences and
#'   `"drop"` removes them.
#' @return The treated vector; the attribute `"outliers"` holds the
#'   positions that were outliers in the input.
#' @export
treat.outliers <- function(x, method = "iqr", k = NULL, action = c("flag", "na", "clip", "drop")) {
  action <- match.arg(action)
  is.out <- flag.outliers(x, method = method, k = k)
  fences <- outlier.fences(x, method = method, k = k)
  out <- switch(action,
    flag = x,
    na = replace(x, is.out, NA),
    clip = as.vector(winsorize(x, fences[["lower"]], fences[["upper"]])),
    drop = x[!is.out]
  )
  attr(out, "outliers") <- which(is.out)
  out
}

#' Report outliers in all numeric columns
#'
#' @param data A data frame to clean.
#' @param method,k Passed to [outlier.fences()].
#' @return A data frame with one row per numeric column and columns
#'   `column`, `n`, `n.missing`, `lower`, `upper`, `n.low` and `n.high`.
#' @export
outlier.report <- function(data, method = "iqr", k = NULL) {
  numeric.cols <- names(data)[vapply(data, is.numeric, logical(1))]
  rows <- lapply(numeric.cols, function(col) {
    x <- data[[col]]
    fences <- outlier.fences(x, method = method, k = k)
    data.frame(
      column = col,
      n = length(x),
      n.missing = sum(is.na(x)),
      lower = fences[["lower"]],
      upper = fences[["upper"]],
      n.low = sum(x < fences[["lower"]], na.rm = TRUE),
      n.high = sum(x > fences[["upper"]], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

#' Clean the numeric columns of a data frame
#'
#' @param data A data frame to clean.
#' @param columns Numeric columns to treat; defaults to all of them.
#' @param method,k Passed to [outlier.fences()].
#' @param action One of `"na"` or `"clip"`, passed to [treat.outliers()].
#' @return A list with elements `data` and `treated` (a named integer
#'   vector with the number of outliers per column).
#' @export
clean.numeric.columns <- function(data, columns = NULL, method = "iqr", k = NULL,
                                  action = c("na", "clip")) {
  action <- match.arg(action)
  if (is.null(columns)) {
    columns <- names(data)[vapply(data, is.numeric, logical(1))]
  }
  unknown <- setdiff(columns, names(data))
  if (length(unknown) > 0L) {
    stop("`data` lacks column(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  treated <- integer()
  for (col in columns) {
    value <- treat.outliers(data[[col]], method = method, k = k, action = action)
    treated[[col]] <- length(attr(value, "outliers"))
    attr(value, "outliers") <- NULL
    data[[col]] <- value
  }
  list(data = data, treated = treated)
}

#' Clean implausible vital signs
#'
#' @description
#' Sets physiologically implausible values to `NA` in the vital sign
#' columns `heart.rate`, `systolic`, `diastolic`, `temperature` and
#' `weight`, using the limits in `limits`. Temperatures recorded in
#' Fahrenheit are converted to Celsius first.
#'
#' @param data A data frame to clean.
#' @param limits Named list of `c(lower, upper)` limits, one per column.
#' @return A list with elements `data` and `removed` (a named integer
#'   vector with the number of values removed per column).
#' @export
clean.vitals <- function(data,
                         limits = list(heart.rate = c(25, 250), systolic = c(50, 260),
                                       diastolic = c(20, 160), temperature = c(30, 44),
                                       weight = c(0.3, 400))) {
  removed <- integer()

  # Clean `heart.rate`
  out <- data$heart.rate < limits$heart.rate[1] | data$heart.rate > limits$heart.rate[2]
  removed[["heart.rate"]] <- sum(out, na.rm = TRUE)
  data$heart.rate[which(out)] <- NA

  # Clean `systolic`
  out <- data$systolic < limits$systolic[1] | data$systolic > limits$systolic[2]
  removed[["systolic"]] <- sum(out, na.rm = TRUE)
  data$systolic[which(out)] <- NA

  # Clean `diastolic`, which must also stay below `systolic`
  out <- data$diastolic < limits$diastolic[1] | data$diastolic > limits$diastolic[2] |
    data$diastolic >= data$systolic
  removed[["diastolic"]] <- sum(out, na.rm = TRUE)
  data$diastolic[which(out)] <- NA

  # Clean `temperature`, converting Fahrenheit readings first
  fahrenheit <- which(data$temperature > 90 & data$temperature < 112)
  data$temperature[fahrenheit] <- (data$temperature[fahrenheit] - 32) * 5 / 9
  out <- data$temperature < limits$temperature[1] | data$temperature > limits$temperature[2]
  removed[["temperature"]] <- sum(out, na.rm = TRUE)
  data$temperature[which(out)] <- NA

  # Clean `weight`
  out <- data$weight < limits$weight[1] | data$weight > limits$weight[2]
  removed[["weight"]] <- sum(out, na.rm = TRUE)
  data$weight[which(out)] <- NA

  list(data = data, removed = removed)
}
