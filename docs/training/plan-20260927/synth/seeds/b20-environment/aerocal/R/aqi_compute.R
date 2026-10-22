#' Truncate concentrations to table precision
#'
#' Concentrations are truncated, not rounded, to the number of decimal places
#' used by the breakpoint table before any lookup.
#'
#' @param conc A numeric vector of pollutant concentrations.
#' @param digits Number of decimal places to keep.
#' @return A numeric vector the same length as `conc`.
#' @keywords internal
truncateConc <- function(conc, digits) {
  scale <- 10^digits
  return(trunc(conc * scale) / scale)
}

#' Compute a pollutant sub-index
#'
#' Locates each concentration in the breakpoint table and interpolates the
#' index linearly within its band.
#'
#' @param conc A numeric vector of pollutant concentrations.
#' @param pollutant Short pollutant code passed to [getBreakpoints()].
#' @return An integer vector of sub-index values; `NA` for missing or
#'   out-of-range concentrations.
#' @family index calculators
#' @export
subIndex <- function(conc, pollutant) {
  tbl <- getBreakpoints(pollutant)
  conc <- truncateConc(conc, attr(tbl, "digits"))
  out <- rep(NA_integer_, length(conc))
  for (i in seq_along(conc)) {
    c0 <- conc[i]
    if (is.na(c0) || c0 < 0) {
      next
    }
    band <- which(c0 >= tbl$concLo & c0 <= tbl$concHi)
    if (length(band) == 0) {
      next
    }
    b <- tbl[band[1], ]
    value <- (b$indexHi - b$indexLo) / (b$concHi - b$concLo) * (c0 - b$concLo) + b$indexLo
    out[i] <- as.integer(round(value))
  }
  return(out)
}

#' Convert a sub-index back to a concentration
#'
#' Inverts the interpolation of [subIndex()], which is useful when reporting
#' the concentration that corresponds to an index target.
#'
#' @param index A numeric vector of index values.
#' @param pollutant Short pollutant code passed to [getBreakpoints()].
#' @return A numeric vector of concentrations; `NA` for out-of-range indices.
#' @family index calculators
#' @export
indexToConc <- function(index, pollutant) {
  tbl <- getBreakpoints(pollutant)
  out <- rep(NA_real_, length(index))
  for (i in seq_along(index)) {
    v <- index[i]
    if (is.na(v) || v < 0) {
      next
    }
    band <- which(v >= tbl$indexLo & v <= tbl$indexHi)
    if (length(band) == 0) {
      next
    }
    b <- tbl[band[1], ]
    value <- (b$concHi - b$concLo) / (b$indexHi - b$indexLo) * (v - b$indexLo) + b$concLo
    out[i] <- truncateConc(value, attr(tbl, "digits"))
  }
  return(out)
}

#' Category label for an index value
#'
#' @param index A numeric vector of index values.
#' @return A character vector of category labels.
#' @export
indexCategory <- function(index) {
  levels <- aqiLevels()
  pos <- findInterval(index, levels$indexLo)
  pos[pos == 0] <- NA_integer_
  return(levels$category[pos])
}

#' Hourly-weighted recent average for particulate matter
#'
#' Computes a weighted average of the most recent 12 hourly concentrations
#' in which the weight decays faster when concentrations change rapidly.
#'
#' @param hourly A numeric vector of hourly concentrations, oldest first.
#' @param minWeight Lower bound for the decay weight.
#' @param minRecent Minimum number of non-missing values among the three most
#'   recent hours.
#' @return A single numeric value, or `NA` when too little recent data exist.
#' @export
weightedRecentAvg <- function(hourly, minWeight = 0.5, minRecent = 2L) {
  recent <- rev(utils::tail(hourly, 12))
  if (sum(!is.na(recent[1:3])) < minRecent) {
    return(NA_real_)
  }
  range <- max(recent, na.rm = TRUE) - min(recent, na.rm = TRUE)
  peak <- max(recent, na.rm = TRUE)
  w <- if (peak > 0) 1 - range / peak else 1
  w <- max(w, minWeight)
  powers <- w^(seq_along(recent) - 1)
  keep <- !is.na(recent)
  return(sum(powers[keep] * recent[keep]) / sum(powers[keep]))
}

#' Overall index from several pollutants
#'
#' Computes each pollutant's sub-index and reports the maximum together with
#' the responsible pollutant.
#'
#' @param data A data frame with one row per site and time and one
#'   concentration column per pollutant.
#' @param pollutants Character vector of pollutant columns to include.
#' @return `data` with added columns `aqi`, `dominant` and `category`.
#' @export
overallIndex <- function(data, pollutants = c("pm25", "pm10", "o3", "no2")) {
  pollutants <- intersect(pollutants, names(data))
  if (length(pollutants) == 0) {
    stop("No pollutant columns found in `data`.", call. = FALSE)
  }
  subs <- lapply(pollutants, \(p) subIndex(data[[p]], p)) |>
    do.call(what = cbind)
  colnames(subs) <- pollutants
  allMissing <- rowSums(!is.na(subs)) == 0
  subs[is.na(subs)] <- -1L
  best <- max.col(subs, ties.method = "first")
  data$aqi <- subs[cbind(seq_len(nrow(subs)), best)]
  data$dominant <- pollutants[best]
  data$aqi[allMissing] <- NA_integer_
  data$dominant[allMissing] <- NA_character_
  data$category <- indexCategory(data$aqi)
  return(data)
}

#' Count days per index category
#'
#' Tallies daily index values into categories for each site, suitable for
#' annual air-quality reporting.
#'
#' @param data A data frame with one row per site and time and one
#'   concentration column per pollutant.
#' @param siteCol Name of the site identifier column.
#' @return A data frame with one row per site and one count column per
#'   category, plus `nValid`.
#' @export
categoryDays <- function(data, siteCol = "site") {
  if (!"aqi" %in% names(data)) {
    data <- overallIndex(data)
  }
  levels <- aqiLevels()$category
  sites <- unique(data[[siteCol]])
  counts <- matrix(0L, nrow = length(sites), ncol = length(levels))
  for (i in seq_along(sites)) {
    cats <- data$category[data[[siteCol]] == sites[i]]
    counts[i, ] <- tabulate(match(cats, levels), nbins = length(levels))
  }
  colnames(counts) <- make.names(levels)
  out <- data.frame(site = sites, counts, nValid = rowSums(counts))
  return(out)
}

#' Ozone index from 8-hour and 1-hour averages
#'
#' Uses the 8-hour average where it falls inside the 8-hour table and the
#' 1-hour value otherwise, reporting the larger index when both apply.
#'
#' @param hourly A numeric vector of hourly concentrations, oldest first.
#' @param oneHourBands A data frame with columns `concLo`, `concHi`,
#'   `indexLo` and `indexHi` for 1-hour ozone in parts per billion.
#' @param minHours Minimum number of valid hours in each 8-hour window.
#' @return A data frame with one row per hour and columns `o3_8h`, `index8h`,
#'   `index1h` and `index`.
#' @export
ozoneIndex <- function(hourly,
                       oneHourBands = data.frame(
                         concLo = c(125, 165, 205, 405),
                         concHi = c(164, 204, 404, 604),
                         indexLo = c(101, 151, 201, 301),
                         indexHi = c(150, 200, 300, 500)
                       ),
                       minHours = 6L) {
  n <- length(hourly)
  avg8 <- rep(NA_real_, n)
  index1h <- rep(NA_integer_, n)
  for (i in seq_len(n)) {
    if (i >= 8) {
      window <- hourly[(i - 7):i]
      if (sum(!is.na(window)) >= minHours) {
        avg8[i] <- mean(window, na.rm = TRUE)
      }
    }
    c1 <- trunc(hourly[i])
    if (is.na(c1)) {
      next
    }
    band <- which(c1 >= oneHourBands$concLo & c1 <= oneHourBands$concHi)
    if (length(band) == 1) {
      b <- oneHourBands[band, ]
      v <- (b$indexHi - b$indexLo) / (b$concHi - b$concLo) * (c1 - b$concLo) + b$indexLo
      index1h[i] <- as.integer(round(v))
    }
  }
  index8h <- subIndex(avg8, "o3")
  index <- pmax(index8h, index1h, na.rm = TRUE)
  return(data.frame(
    o3_8h = avg8,
    index8h = index8h,
    index1h = index1h,
    index = index
  ))
}
