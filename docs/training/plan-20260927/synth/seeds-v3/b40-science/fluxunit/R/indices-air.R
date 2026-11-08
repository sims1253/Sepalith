#' Breakpoint table for an air quality sub-index
#'
#' Each pollutant has seven concentration cut points that bound the six
#' index bands 0-50, 51-100, 101-150, 151-200, 201-300 and 301-500.
#'
#' @param pollutant One of `"pm25"`, `"pm10"`, `"o3"`, `"no2"`, `"so2"`
#'   or `"co"`.
#' @return A data frame with columns `conc_lo`, `conc_hi`, `index_lo` and
#'   `index_hi`, one row per band.
#' @keywords internal
aqi_breakpoints <- function(pollutant = c("pm25", "pm10", "o3", "no2", "so2", "co")) {
  pollutant <- match.arg(pollutant)
  index_cuts <- c(0, 50, 100, 150, 200, 300, 500)
  conc_cuts <- switch(pollutant,
    # ug/m3, 24 hour mean
    pm25 = c(0, 12, 35.4, 55.4, 150.4, 250.4, 500.4),
    # ug/m3, 24 hour mean
    pm10 = c(0, 54, 154, 254, 354, 424, 604),
    # ppb, 8 hour mean
    o3 = c(0, 54, 70, 85, 105, 200, 400),
    # ppb, 1 hour mean
    no2 = c(0, 53, 100, 360, 649, 1249, 2049),
    # ppb, 1 hour mean
    so2 = c(0, 35, 75, 185, 304, 604, 1004),
    # ppm, 8 hour mean
    co = c(0, 4.4, 9.4, 12.4, 15.4, 30.4, 50.4)
  )
  n_bands <- length(index_cuts) - 1L
  data.frame(
    conc_lo = conc_cuts[seq_len(n_bands)],
    conc_hi = conc_cuts[-1],
    index_lo = c(0, index_cuts[2:n_bands] + 1),
    index_hi = index_cuts[-1]
  )
}

#' Sub-index for one pollutant
#'
#' Interpolates linearly inside the band that contains each concentration.
#' Concentrations above the last cut point are capped at an index of 500.
#'
#' @param conc A numeric vector of concentrations.
#' @param pollutant Passed to [aqi_breakpoints()].
#' @return A numeric vector of sub-index values, rounded to integers.
#' @keywords internal
aqi_subindex <- function(conc, pollutant) {
  bp <- aqi_breakpoints(pollutant)
  band <- findInterval(conc, c(bp$conc_lo, Inf), rightmost.closed = TRUE)
  band[band > nrow(bp)] <- nrow(bp)
  out <- rep(NA_real_, length(conc))
  ok <- !is.na(conc) & band > 0L
  b <- band[ok]
  frac <- (conc[ok] - bp$conc_lo[b]) / (bp$conc_hi[b] - bp$conc_lo[b])
  out[ok] <- bp$index_lo[b] + pmin(frac, 1) * (bp$index_hi[b] - bp$index_lo[b])
  round(out)
}

#' Validate a concentration vector
#'
#' @param x A numeric vector of concentrations, or `NULL`.
#' @param name The argument name used in messages.
#' @param n The expected length.
#' @return `x`, invisibly, if it is valid.
#' @noRd
check_concentration <- function(x, name, n) {
  if (!is.numeric(x)) {
    stop(sprintf("`%s` must be numeric.", name), call. = FALSE)
  }
  if (length(x) != n) {
    stop(sprintf("`%s` must have length %d, not %d.", name, n, length(x)), call. = FALSE)
  }
  if (any(x < 0, na.rm = TRUE)) {
    stop(sprintf("`%s` must not contain negative concentrations.", name), call. = FALSE)
  }
  invisible(x)
}

#' Air quality index from pollutant concentrations
#'
#' Computes one sub-index per supplied pollutant and takes the maximum as
#' the overall index. Pollutants left as `NULL` are skipped.
#'
#' @param pm25 Fine particulate matter, ug/m3.
#' @param pm10 Coarse particulate matter, ug/m3.
#' @param o3 Ozone, ppb.
#' @param no2 Nitrogen dioxide, ppb.
#' @param so2 Sulphur dioxide, ppb.
#' @return A data frame with columns `sub_pm25`, `sub_pm10`, `sub_o3`,
#'   `sub_no2`, `sub_so2`, `aqi` and `dominant`.
#' @export
air_quality_index <- function(pm25 = NULL, pm10 = NULL, o3 = NULL, no2 = NULL, so2 = NULL) {
  supplied <- Filter(Negate(is.null), list(pm25 = pm25, pm10 = pm10, o3 = o3,
                                           no2 = no2, so2 = so2))
  if (length(supplied) == 0L) {
    stop("Supply at least one pollutant.", call. = FALSE)
  }
  n <- length(supplied[[1]])

  # PM2.5 sub-index
  sub_pm25 <- rep(NA_real_, n)
  if (!is.null(pm25)) {
    check_concentration(pm25, "pm25", n)
    sub_pm25 <- aqi_subindex(pm25, "pm25")
  }

  # PM10 sub-index
  sub_pm10 <- rep(NA_real_, n)
  if (!is.null(pm10)) {
    check_concentration(pm10, "pm10", n)
    sub_pm10 <- aqi_subindex(pm10, "pm10")
  }

  # O3 sub-index
  sub_o3 <- rep(NA_real_, n)
  if (!is.null(o3)) {
    check_concentration(o3, "o3", n)
    sub_o3 <- aqi_subindex(o3, "o3")
  }

  # NO2 sub-index
  sub_no2 <- rep(NA_real_, n)
  if (!is.null(no2)) {
    check_concentration(no2, "no2", n)
    sub_no2 <- aqi_subindex(no2, "no2")
  }

  # SO2 sub-index
  sub_so2 <- rep(NA_real_, n)
  if (!is.null(so2)) {
    check_concentration(so2, "so2", n)
    sub_so2 <- aqi_subindex(so2, "so2")
  }

  subs <- cbind(pm25 = sub_pm25, pm10 = sub_pm10, o3 = sub_o3, no2 = sub_no2, so2 = sub_so2)
  all_na <- rowSums(!is.na(subs)) == 0L
  aqi <- ifelse(all_na, NA_real_, suppressWarnings(apply(subs, 1, max, na.rm = TRUE)))
  dominant <- ifelse(all_na, NA_character_,
                     colnames(subs)[apply(replace(subs, is.na(subs), -1), 1, which.max)])
  data.frame(
    sub_pm25 = sub_pm25, sub_pm10 = sub_pm10, sub_o3 = sub_o3,
    sub_no2 = sub_no2, sub_so2 = sub_so2, aqi = aqi, dominant = dominant
  )
}

#' Outdoor comfort index
#'
#' Scores temperature, humidity and wind separately against an optimum
#' range and combines the three scores with weights.
#'
#' @details Sub-indices:
#'   * `score_temp`: 1 inside `optimum$temp`, falling to 0 at
#'     `tolerance[["temp"]]` degrees outside it.
#'   * `score_humidity`: the same for relative humidity in percent.
#'   * `score_wind`: the same for wind speed in m/s.
#'
#' @param temp Air temperature in degrees Celsius.
#' @param humidity Relative humidity in percent.
#' @param wind Wind speed in m/s.
#' @param weights Named weights for `temp`, `humidity` and `wind`.
#' @return A data frame with columns `score_temp`, `score_humidity`,
#'   `score_wind` and `comfort`.
#' @export
comfort_index <- function(temp, humidity, wind,
                          weights = c(temp = 0.5, humidity = 0.3, wind = 0.2)) {
  optimum <- list(temp = c(18, 24), humidity = c(40, 60), wind = c(0, 3))
  tolerance <- c(temp = 10, humidity = 30, wind = 7)
  weights <- weights[names(optimum)] / sum(weights[names(optimum)])

  # Score `temp`
  lo <- optimum$temp[1]
  hi <- optimum$temp[2]
  gap <- pmax(lo - temp, temp - hi, 0)
  score_temp <- 1 - pmin(gap / tolerance[["temp"]], 1)

  # Score `humidity`
  lo <- optimum$humidity[1]
  hi <- optimum$humidity[2]
  gap <- pmax(lo - humidity, humidity - hi, 0)
  score_humidity <- 1 - pmin(gap / tolerance[["humidity"]], 1)

  # Score `wind`
  lo <- optimum$wind[1]
  hi <- optimum$wind[2]
  gap <- pmax(lo - wind, wind - hi, 0)
  score_wind <- 1 - pmin(gap / tolerance[["wind"]], 1)

  comfort <- weights[["temp"]] * score_temp +
    weights[["humidity"]] * score_humidity +
    weights[["wind"]] * score_wind
  data.frame(
    score_temp = score_temp,
    score_humidity = score_humidity,
    score_wind = score_wind,
    comfort = comfort
  )
}
