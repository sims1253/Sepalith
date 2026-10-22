#' Build a breakpoint table
#'
#' Assembles concentration and index breakpoints into the standard table
#' layout used throughout the package.
#'
#' @param pollutant Short pollutant code, for example `"pm25"`.
#' @param unit Concentration unit as a string.
#' @param concLo,concHi Numeric vectors of lower and upper concentration
#'   bounds for each category.
#' @param digits Number of decimal places to which concentrations are
#'   truncated before lookup.
#' @return A data frame of class `aq_breakpoints` with one row per category.
#' @keywords internal
newBreakpoints <- function(pollutant, unit, concLo, concHi, digits) {
  if (length(concLo) != length(concHi)) {
    stop("`concLo` and `concHi` must have the same length.", call. = FALSE)
  }
  levels <- aqiLevels()
  n <- length(concLo)
  tbl <- data.frame(
    pollutant = pollutant,
    unit = unit,
    concLo = concLo,
    concHi = concHi,
    indexLo = levels$indexLo[seq_len(n)],
    indexHi = levels$indexHi[seq_len(n)],
    category = levels$category[seq_len(n)],
    stringsAsFactors = FALSE
  )
  attr(tbl, "digits") <- digits
  class(tbl) <- c("aq_breakpoints", class(tbl))
  return(tbl)
}

#' Index levels and category labels
#'
#' The index bands shared by every pollutant table.
#'
#' @return A data frame with columns `indexLo`, `indexHi`, `category` and
#'   `colour`.
#' @export
aqiLevels <- function() {
  return(data.frame(
    indexLo = c(0, 51, 101, 151, 201, 301),
    indexHi = c(50, 100, 150, 200, 300, 500),
    category = c(
      "Good", "Moderate", "Unhealthy for sensitive groups",
      "Unhealthy", "Very unhealthy", "Hazardous"
    ),
    colour = c("#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023"),
    stringsAsFactors = FALSE
  ))
}

#' Breakpoints for fine particulate matter
#'
#' 24-hour average PM2.5 breakpoints in micrograms per cubic metre.
#'
#' @return A data frame of class `aq_breakpoints`.
#' @family breakpoint tables
#' @export
breakpointsPm25 <- function() {
  return(newBreakpoints(
    pollutant = "pm25",
    unit = "ug/m3",
    concLo = c(0.0, 9.1, 35.5, 55.5, 125.5, 225.5),
    concHi = c(9.0, 35.4, 55.4, 125.4, 225.4, 500.4),
    digits = 1
  ))
}

#' Breakpoints for coarse particulate matter
#'
#' 24-hour average PM10 breakpoints in micrograms per cubic metre.
#'
#' @return A data frame of class `aq_breakpoints`.
#' @family breakpoint tables
#' @export
breakpointsPm10 <- function() {
  return(newBreakpoints(
    pollutant = "pm10",
    unit = "ug/m3",
    concLo = c(0, 55, 155, 255, 355, 425),
    concHi = c(54, 154, 254, 354, 424, 604),
    digits = 0
  ))
}

#' Breakpoints for ozone
#'
#' 8-hour average ozone breakpoints in parts per billion. Values above the
#' last 8-hour band should be assessed with 1-hour averages instead.
#'
#' @return A data frame of class `aq_breakpoints`.
#' @family breakpoint tables
#' @export
breakpointsO3 <- function() {
  return(newBreakpoints(
    pollutant = "o3",
    unit = "ppb",
    concLo = c(0, 55, 71, 86, 106),
    concHi = c(54, 70, 85, 105, 200),
    digits = 0
  ))
}

#' Breakpoints for nitrogen dioxide
#'
#' 1-hour average NO2 breakpoints in parts per billion.
#'
#' @return A data frame of class `aq_breakpoints`.
#' @family breakpoint tables
#' @export
breakpointsNo2 <- function() {
  return(newBreakpoints(
    pollutant = "no2",
    unit = "ppb",
    concLo = c(0, 54, 101, 361, 650, 1250),
    concHi = c(53, 100, 360, 649, 1249, 2049),
    digits = 0
  ))
}

#' Look up the breakpoint table for a pollutant
#'
#' @param pollutant Short pollutant code, one of `"pm25"`, `"pm10"`, `"o3"`
#'   and `"no2"`.
#' @return A data frame of class `aq_breakpoints`.
#' @export
getBreakpoints <- function(pollutant = c("pm25", "pm10", "o3", "no2")) {
  pollutant <- match.arg(pollutant)
  tbl <- switch(pollutant,
    pm25 = breakpointsPm25(),
    pm10 = breakpointsPm10(),
    o3 = breakpointsO3(),
    no2 = breakpointsNo2()
  )
  return(tbl)
}

#' Validate a breakpoint table
#'
#' Checks that concentration bands are increasing, non-overlapping and
#' contiguous to within the table's precision.
#'
#' @param tbl A data frame of class `aq_breakpoints`.
#' @return `TRUE`, invisibly, or an error describing the first problem found.
#' @export
validateBreakpoints <- function(tbl) {
  if (!inherits(tbl, "aq_breakpoints")) {
    stop("`tbl` must be an `aq_breakpoints` table.", call. = FALSE)
  }
  step <- 10^(-attr(tbl, "digits"))
  if (any(tbl$concHi < tbl$concLo)) {
    stop("A band has its upper bound below its lower bound.", call. = FALSE)
  }
  for (i in seq_len(nrow(tbl))[-1]) {
    gap <- tbl$concLo[i] - tbl$concHi[i - 1]
    if (gap <= 0) {
      stop(sprintf("Bands %d and %d overlap.", i - 1, i), call. = FALSE)
    } else if (abs(gap - step) > step / 2) {
      stop(sprintf("Bands %d and %d are not contiguous.", i - 1, i), call. = FALSE)
    }
  }
  return(invisible(TRUE))
}

#' Revise one band of a breakpoint table
#'
#' Replaces the concentration bounds of a single category and shifts the
#' neighbouring bands so that the table stays contiguous.
#'
#' @param tbl A data frame of class `aq_breakpoints`.
#' @param category Category label of the band to revise.
#' @param concLo,concHi New lower and upper concentration bounds.
#' @return A revised data frame of class `aq_breakpoints`.
#' @export
reviseBreakpoints <- function(tbl, category, concLo, concHi) {
  validateBreakpoints(tbl)
  i <- match(category, tbl$category)
  if (is.na(i)) {
    stop("Unknown category: ", category, call. = FALSE)
  }
  if (concHi <= concLo) {
    stop("`concHi` must exceed `concLo`.", call. = FALSE)
  }
  step <- 10^(-attr(tbl, "digits"))
  tbl$concLo[i] <- concLo
  tbl$concHi[i] <- concHi
  if (i > 1) {
    tbl$concHi[i - 1] <- concLo - step
    if (tbl$concHi[i - 1] <= tbl$concLo[i - 1]) {
      stop("Revision would collapse the band below.", call. = FALSE)
    }
  }
  if (i < nrow(tbl)) {
    tbl$concLo[i + 1] <- concHi + step
    if (tbl$concLo[i + 1] >= tbl$concHi[i + 1]) {
      stop("Revision would collapse the band above.", call. = FALSE)
    }
  }
  validateBreakpoints(tbl)
  return(tbl)
}
