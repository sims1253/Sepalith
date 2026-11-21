#' Summary statistics per site
#'
#' @param data A data frame of observations, one row per reading.
#' @param value Name of the measurement column.
#' @param site Name of the site identifier column.
#' @param stats Statistics to compute, any of `"n"`, `"mean"`, `"sd"`,
#'   `"min"`, `"median"` and `"max"`.
#' @export
#' @return A data frame with a `site` column followed by one column per
#'   element of `stats`, in the order `n`, `mean`, `sd`, `min`, `median`,
#'   `max`.
site.summary <- function(data, value, site = "site",
                         stats = c("n", "mean", "sd", "min", "median", "max")) {
  stats <- match.arg(stats, several.ok = TRUE)
  x <- data[[value]]
  g <- factor(data[[site]])
  out <- data.frame(site = levels(g))

  if ("n" %in% stats) {
    out$n <- as.integer(tapply(x, g, function(v) sum(!is.na(v))))
  }
  if ("mean" %in% stats) {
    out$mean <- as.numeric(tapply(x, g, mean, na.rm = TRUE))
  }
  if ("sd" %in% stats) {
    out$sd <- as.numeric(tapply(x, g, stats::sd, na.rm = TRUE))
  }
  if ("min" %in% stats) {
    out$min <- as.numeric(tapply(x, g, min, na.rm = TRUE))
  }
  if ("median" %in% stats) {
    out$median <- as.numeric(tapply(x, g, stats::median, na.rm = TRUE))
  }
  if ("max" %in% stats) {
    out$max <- as.numeric(tapply(x, g, max, na.rm = TRUE))
  }
  out
}

#' Data completeness per site
#'
#' @inheritParams site.summary
#' @param time Name of the `POSIXct` time column.
#' @param interval Expected spacing of readings in seconds.
#' @export
#' @return A data frame with columns `site`, `first`, `last`, `expected`,
#'   `received`, `valid`, `completeness` (percent of expected readings
#'   that are valid) and `max_gap_h` (longest gap between readings, in
#'   hours).
site.completeness <- function(data, value, site = "site", time = "time", interval = 3600) {
  sites <- sort(unique(data[[site]]))
  rows <- vector("list", length(sites))
  for (i in seq_along(sites)) {
    sub <- data[data[[site]] == sites[i], , drop = FALSE]
    t <- sub[[time]]
    first <- min(t, na.rm = TRUE)
    last <- max(t, na.rm = TRUE)
    expected <- as.integer(floor(as.numeric(difftime(last, first, units = "secs")) /
                                   interval)) + 1L
    received <- nrow(sub)
    valid <- sum(!is.na(sub[[value]]))
    gaps <- diff(sort(as.numeric(t)))
    max.gap <- if (length(gaps) > 0L) max(gaps) / 3600 else NA_real_

    rows[[i]] <- data.frame(
      site = sites[i], first = first, last = last, expected = expected,
      received = received, valid = valid, completeness = 100 * valid / expected,
      max_gap_h = max.gap
    )
  }
  do.call(rbind, rows)
}

#' Extreme readings per site
#'
#' For each site, finds the lowest and the highest reading together with
#' the time at which it occurred.
#'
#' @inheritParams site.completeness
#' @export
#' @return A data frame with columns `site`, `min`, `min_time`, `max` and
#'   `max_time`.
site.extremes <- function(data, value, site = "site", time = "time") {
  sites <- sort(unique(data[[site]]))
  out <- data.frame(site = sites, min = NA_real_, min_time = data[[time]][NA_integer_],
                    max = NA_real_, max_time = data[[time]][NA_integer_])
  for (i in seq_along(sites)) {
    sub <- data[data[[site]] == sites[i], , drop = FALSE]
    x <- sub[[value]]
    if (all(is.na(x))) {
      next
    }

    # Lowest reading
    lo <- which.min(x)
    out$min[i] <- x[lo]
    out$min_time[i] <- sub[[time]][lo]

    # Highest reading
    hi <- which.max(x)
    out$max[i] <- x[hi]
    out$max_time[i] <- sub[[time]][hi]
  }
  out
}

#' Count threshold exceedances per site
#'
#' @inheritParams site.summary
#' @param thresholds A named numeric vector of thresholds, for example
#'   `c(alert = 50, alarm = 100)`.
#' @param direction `"above"` counts readings greater than each threshold,
#'   `"below"` readings smaller than it.
#' @export
#' @return A data frame with columns `site` and `n_valid`, then for
#'   each threshold a count column `n_<threshold name>` and a percentage
#'   column `pct_<threshold name>`.
site.exceedances <- function(data, value, thresholds, site = "site",
                             direction = c("above", "below")) {
  direction <- match.arg(direction)
  if (!value %in% names(data)) {
    stop(sprintf("`data` has no column `%s`.", value), call. = FALSE)
  }
  if (is.null(names(thresholds)) || any(names(thresholds) == "")) {
    stop("`thresholds` must be a named vector.", call. = FALSE)
  }
  x <- data[[value]]
  g <- factor(data[[site]])
  out <- data.frame(site = levels(g))
  out$n_valid <- as.integer(tapply(!is.na(x), g, sum))

  for (nm in names(thresholds)) {
    hit <- if (direction == "above") x > thresholds[[nm]] else x < thresholds[[nm]]
    count <- as.integer(tapply(hit, g, sum, na.rm = TRUE))
    out[[paste0("n_", nm)]] <- count
    out[[paste0("pct_", nm)]] <- ifelse(out$n_valid > 0L, 100 * count / out$n_valid, NA_real_)
  }
  out
}

#' Seasonal means per site
#'
#' @inheritParams site.completeness
#' @param hemisphere `"north"` or `"south"`; only changes the column
#'   names of the result.
#' @export
#' @return A data frame with columns `site`, `DJF`, `MAM`, `JJA` and
#'   `SON` holding mean values; in the southern hemisphere the same
#'   columns are named after the local season (`summer`, `autumn`,
#'   `winter`, `spring`) in that order.
site.seasonal <- function(data, value, site = "site", time = "time",
                          hemisphere = c("north", "south")) {
  hemisphere <- match.arg(hemisphere)
  month <- as.integer(format(data[[time]], "%m"))
  season <- c("DJF", "DJF", "MAM", "MAM", "MAM", "JJA",
              "JJA", "JJA", "SON", "SON", "SON", "DJF")[month]
  season <- factor(season, levels = c("DJF", "MAM", "JJA", "SON"))
  tab <- tapply(data[[value]], list(data[[site]], season), mean, na.rm = TRUE)
  out <- data.frame(site = rownames(tab), tab, row.names = NULL, check.names = FALSE)
  if (hemisphere == "south") {
    names(out)[-1] <- c("summer", "autumn", "winter", "spring")
  }
  out
}
