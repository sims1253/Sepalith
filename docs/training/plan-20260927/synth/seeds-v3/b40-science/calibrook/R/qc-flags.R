#' Translate numeric flags into labels
#'
#' @description
#' Flags use `0` for pass, `1` for fail and `NA` for not checked.
#'
#' @param flag An integer vector of flags.
#' @param scheme Label set: `"binary"` (`pass`/`fail`/`missing`),
#'   `"traffic"` (`green`/`red`/`grey`), `"letter"` (`P`/`F`/`M`) or
#'   `"symbol"` (`.`/`x`/`?`).
#' @return A character vector the same length as `flag`.
#' @export
flagLabels = function(flag, scheme = c("binary", "traffic", "letter", "symbol")) {
  scheme = match.arg(scheme)
  labels = switch(scheme,
    binary = c(pass = "pass", fail = "fail", missing = "missing"),
    traffic = c(pass = "green", fail = "red", missing = "grey"),
    letter = c(pass = "P", fail = "F", missing = "M"),
    symbol = c(pass = ".", fail = "x", missing = "?")
  )
  out = ifelse(flag == 0L, labels[["pass"]], labels[["fail"]])
  out[is.na(flag)] = labels[["missing"]]
  return(unname(out))
}

#' Combine several flag vectors into one
#'
#' @description
#' All inputs must have the same length. Missing flags are ignored unless
#' every flag at a position is missing.
#'
#' @param ... Integer flag vectors, for example from range, step and
#'   persistence checks.
#' @param rule `"worst"` fails when any check fails, `"majority"` when
#'   more than half fail, `"all"` only when every check fails and
#'   `"first"` uses the first non-missing flag.
#' @return An integer flag vector.
#' @export
combineFlags = function(..., rule = c("worst", "majority", "all", "first")) {
  rule = match.arg(rule)
  m = cbind(...)
  if (ncol(m) == 0L) {
    stop("Supply at least one flag vector.", call. = FALSE)
  }
  nChecked = rowSums(!is.na(m))
  nFailed = rowSums(m == 1L, na.rm = TRUE)
  out = switch(rule,
    worst = {
      as.integer(nFailed > 0L)
    },
    majority = {
      as.integer(nFailed > nChecked / 2)
    },
    all = {
      as.integer(nFailed == nChecked)
    },
    first = {
      firstCol = max.col(!is.na(m), ties.method = "first")
      as.integer(m[cbind(seq_len(nrow(m)), firstCol)])
    }
  )
  out[nChecked == 0L] = NA_integer_
  return(out)
}

#' Cross-variable consistency checks
#'
#' @description
#' Each rule needs two columns of `data` and is skipped when either is
#' absent.
#'
#' @param data A data frame of observations at one station.
#' @param tol Tolerance added to every comparison.
#' @return A data frame with one integer flag column per applied rule:
#'   `dew_above_temp` (`dew_point` vs `temp`), `min_above_max`
#'   (`temp_min` vs `temp_max`), `gust_below_speed` (`gust` vs
#'   `wind_speed`), `rain_when_dry` (`precip` vs `rh`) and
#'   `calm_with_direction` (`wind_speed` vs `wind_dir`).
#' @export
crossCheck = function(data, tol = 0) {
  out = data.frame(row.names = seq_len(nrow(data)))
  has = function(cols) all(cols %in% names(data))

  # Rule `dew_above_temp`
  if (has(c("dew_point", "temp"))) {
    bad = data$dew_point > data$temp + tol
    out$dew_above_temp = as.integer(bad)
  }

  # Rule `min_above_max`
  if (has(c("temp_min", "temp_max"))) {
    bad = data$temp_min > data$temp_max + tol
    out$min_above_max = as.integer(bad)
  }

  # Rule `gust_below_speed`
  if (has(c("gust", "wind_speed"))) {
    bad = data$gust < data$wind_speed - tol
    out$gust_below_speed = as.integer(bad)
  }

  # Rule `rain_when_dry`
  if (has(c("precip", "rh"))) {
    bad = data$precip > 0 & data$rh < 50 - tol
    out$rain_when_dry = as.integer(bad)
  }

  # Rule `calm_with_direction`
  if (has(c("wind_speed", "wind_dir"))) {
    bad = data$wind_speed == 0 & !is.na(data$wind_dir) & data$wind_dir != 0
    out$calm_with_direction = as.integer(bad)
  }

  rownames(out) = NULL
  return(out)
}

#' Count flags per check
#'
#' @param flags A data frame or list of integer flag vectors, one per
#'   check or measure.
#' @return A data frame with columns `check`, `nPass`, `nFail`,
#'   `nMissing`, `pctFail` (percent of checked readings that failed),
#'   `firstFail` and `lastFail` (positions of the first and last failure).
#' @export
flagSummary = function(flags) {
  flags = as.list(flags)
  if (is.null(names(flags)) || any(names(flags) == "")) {
    stop("`flags` must have a name for every check.", call. = FALSE)
  }

  # Counts
  nPass = vapply(flags, function(f) sum(f == 0L, na.rm = TRUE), integer(1))
  nFail = vapply(flags, function(f) sum(f == 1L, na.rm = TRUE), integer(1))
  nMissing = vapply(flags, function(f) sum(is.na(f)), integer(1))
  nChecked = nPass + nFail
  pctFail = ifelse(nChecked > 0L, 100 * nFail / nChecked, NA_real_)

  # Positions
  firstFail = vapply(flags, function(f) {
    hit = which(f == 1L)
    if (length(hit) > 0L) hit[1] else NA_integer_
  }, integer(1))
  lastFail = vapply(flags, function(f) {
    hit = which(f == 1L)
    if (length(hit) > 0L) hit[length(hit)] else NA_integer_
  }, integer(1))

  return(data.frame(
    check = names(flags),
    nPass = unname(nPass),
    nFail = unname(nFail),
    nMissing = unname(nMissing),
    pctFail = unname(pctFail),
    firstFail = unname(firstFail),
    lastFail = unname(lastFail)
  ))
}

#' Replace flagged readings
#'
#' @param data A data frame of observations at one station.
#' @param flags A data frame of integer flags with one column per masked
#'   column of `data`, in the same row order.
#' @param replacement The value to write over failed readings.
#' @return `data` with failed readings replaced and an attribute `masked`
#'   giving the number of replaced readings per column.
#' @export
maskFlagged = function(data, flags, replacement = NA) {
  if (nrow(flags) != nrow(data)) {
    stop("`flags` and `data` must have the same number of rows.", call. = FALSE)
  }
  cols = intersect(names(flags), names(data))
  masked = integer(length(cols))
  names(masked) = cols
  for (col in cols) {
    bad = !is.na(flags[[col]]) & flags[[col]] == 1L
    data[[col]][bad] = replacement
    masked[[col]] = sum(bad)
  }
  attr(data, "masked") = masked
  return(data)
}

#' Find episodes of consecutive failed readings
#'
#' @param flag An integer flag vector in time order.
#' @param time Optional vector of times, same length as `flag`.
#' @return A data frame with columns `start`, `end` and `length`, one row
#'   per episode. `start` and `end` are positions, or times when `time` is
#'   given.
#' @export
flagEpisodes = function(flag, time = NULL) {
  fail = !is.na(flag) & flag == 1L
  runs = rle(fail)
  ends = cumsum(runs$lengths)
  starts = ends - runs$lengths + 1L
  keep = runs$values
  start = starts[keep]
  end = ends[keep]
  if (!is.null(time)) {
    start = time[start]
    end = time[end]
  }
  return(data.frame(start = start, end = end, length = runs$lengths[keep]))
}
