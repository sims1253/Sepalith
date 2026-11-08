#' @title Truncate timestamps to a unit
#' @param x A `POSIXct` vector.
#' @param unit One of `"second"`, `"minute"`, `"hour"`, `"day"`, `"week"`
#'   (starting Monday) or `"month"`.
#' @param tz Time zone in which the boundaries are taken.
#' @return A `POSIXct` vector in `tz`.
#' @rdname time.trunc
#' @export
time.floor <- function(x, unit = c("second", "minute", "hour", "day", "week", "month"),
                       tz = "UTC") {
  unit <- match.arg(unit)
  lt <- as.POSIXlt(x, tz = tz)
  switch(unit,
    second = {
      lt$sec <- floor(lt$sec)
    },
    minute = {
      lt$sec <- 0
    },
    hour = {
      lt$sec <- 0
      lt$min <- 0L
    },
    day = {
      lt$sec <- 0
      lt$min <- 0L
      lt$hour <- 0L
    },
    week = {
      lt$sec <- 0
      lt$min <- 0L
      lt$hour <- 0L
      lt$mday <- lt$mday - (lt$wday + 6L) %% 7L
    },
    month = {
      lt$sec <- 0
      lt$min <- 0L
      lt$hour <- 0L
      lt$mday <- 1L
    }
  )
  as.POSIXct(lt, tz = tz)
}

#' @rdname time.trunc
#' @export
time.ceiling <- function(x, unit = c("second", "minute", "hour", "day", "week", "month"),
                         tz = "UTC") {
  unit <- match.arg(unit)
  lo <- time.floor(x, unit, tz)
  lt <- as.POSIXlt(lo, tz = tz)
  bump <- !is.na(x) & lo < x
  switch(unit,
    second = {
      lt$sec[bump] <- lt$sec[bump] + 1
    },
    minute = {
      lt$min[bump] <- lt$min[bump] + 1L
    },
    hour = {
      lt$hour[bump] <- lt$hour[bump] + 1L
    },
    day = {
      lt$mday[bump] <- lt$mday[bump] + 1L
    },
    week = {
      lt$mday[bump] <- lt$mday[bump] + 7L
    },
    month = {
      lt$mon[bump] <- lt$mon[bump] + 1L
    }
  )
  as.POSIXct(lt, tz = tz)
}

#' @title Round timestamps to the nearest unit
#' @param x A `POSIXct` vector.
#' @param unit One of `"second"`, `"minute"`, `"hour"`, `"day"`, `"week"`
#'   (starting Monday) or `"month"`.
#' @param tz Time zone in which the boundaries are taken.
#' @return A `POSIXct` vector in `tz`; exact halves round up.
#' @export
time.round <- function(x, unit = c("second", "minute", "hour", "day", "week", "month"),
                       tz = "UTC") {
  unit <- match.arg(unit)
  lo <- time.floor(x, unit, tz)
  hi <- time.ceiling(x, unit, tz)
  below <- as.numeric(x) - as.numeric(lo)
  above <- as.numeric(hi) - as.numeric(x)
  out <- lo
  up <- !is.na(x) & above <= below
  out[up] <- hi[up]
  out
}

#' @title Parse timestamps in a named input format
#' @param x A character (or, for the epoch formats, numeric) vector.
#' @param format One of `"iso"` (`2024-03-05 14:30:00`), `"iso.ms"`
#'   (`2024-03-05 14:30:00.250`), `"iso.t"`
#'   (`2024-03-05T14:30:00Z`), `"epoch"` (seconds), `"epoch.ms"`
#'   (milliseconds) or `"http"` (`Tue, 05 Mar 2024 14:30:00 GMT`).
#' @param tz Time zone of the result; text without an offset is read in it.
#' @return A `POSIXct` vector; unparseable values give `NA`.
#' @export
parse.timestamp <- function(x, format = c("iso", "iso.ms", "iso.t", "epoch", "epoch.ms",
                                          "http"),
                            tz = "UTC") {
  format <- match.arg(format)
  switch(format,
    iso = {
      as.POSIXct(x, format = "%Y-%m-%d %H:%M:%S", tz = tz)
    },
    iso.ms = {
      as.POSIXct(x, format = "%Y-%m-%d %H:%M:%OS", tz = tz)
    },
    iso.t = {
      utc <- as.POSIXct(sub("Z$", "", x), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
      as.POSIXct(format(utc, tz = tz, usetz = FALSE), tz = tz)
    },
    epoch = {
      as.POSIXct(as.numeric(x), origin = "1970-01-01", tz = tz)
    },
    epoch.ms = {
      as.POSIXct(as.numeric(x) / 1000, origin = "1970-01-01", tz = tz)
    },
    http = {
      body <- sub("^[A-Za-z]{3}, ", "", sub(" GMT$", "", x))
      months <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
        "Nov", "Dec")
      parts <- regmatches(body, regexec("^(\\d{2}) ([A-Za-z]{3}) (\\d{4}) (.+)$", body))
      iso <- vapply(parts, function(p) {
        if (length(p) != 5L) return(NA_character_)
        sprintf("%s-%02d-%s %s", p[4L], match(p[3L], months), p[2L], p[5L])
      }, character(1))
      utc <- as.POSIXct(iso, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
      as.POSIXct(format(utc, tz = tz, usetz = FALSE), tz = tz)
    }
  )
}

#' @title Show timestamps in another time zone
#' @param x A `POSIXct` vector.
#' @param tz Target time zone.
#' @return `x` with its `tzone` attribute set to `tz`; the instants are
#'   unchanged.
#' @export
time.convert <- function(x, tz) {
  if (!inherits(x, "POSIXct")) {
    stop("`x` must be a `POSIXct` vector.", call. = FALSE)
  }
  attr(x, "tzone") <- tz
  x
}

#' @title Count timestamps per time bucket
#' @param x A `POSIXct` vector.
#' @param unit One of `"second"`, `"minute"`, `"hour"`, `"day"`, `"week"`
#'   (starting Monday) or `"month"`.
#' @param tz Time zone in which the buckets are taken.
#' @param fill If `TRUE`, empty buckets between the first and last one are
#'   included with a count of zero.
#' @return A data frame with columns `bucket` (bucket start), `label`
#'   (formatted with [strftime()]) and `n`.
#' @export
time.bucket <- function(x, unit = c("second", "minute", "hour", "day", "week", "month"),
                        tz = "UTC", fill = FALSE) {
  unit <- match.arg(unit)
  fmt <- switch(unit,
    second = "%Y-%m-%d %H:%M:%S",
    minute = "%Y-%m-%d %H:%M",
    hour = "%Y-%m-%d %H:00",
    day = "%Y-%m-%d",
    week = "%G-W%V",
    month = "%Y-%m"
  )
  starts <- time.floor(x[!is.na(x)], unit, tz)
  counts <- table(as.numeric(starts))
  bucket <- as.POSIXct(as.numeric(names(counts)), origin = "1970-01-01", tz = tz)
  n <- as.integer(counts)

  if (fill && length(bucket) > 1L) {
    all <- seq(min(bucket), max(bucket), by = if (unit == "second") "sec" else unit)
    n <- as.integer(n[match(as.numeric(all), as.numeric(bucket))])
    n[is.na(n)] <- 0L
    bucket <- all
  }
  data.frame(bucket = bucket, label = strftime(bucket, fmt, tz = tz), n = n)
}
