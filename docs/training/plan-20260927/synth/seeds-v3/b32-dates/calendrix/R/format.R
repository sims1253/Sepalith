#' English ordinal suffix for day numbers
#'
#' @param day Integer vector of days of the month.
#' @return A character vector such as `"1st"`, `"22nd"` or `"13th"`.
#' @noRd
ordinal_day <- function(day) {
  day <- as.integer(day)
  suffix <- rep("th", length(day))
  teen <- day %% 100L %in% 11:13
  suffix[!teen & day %% 10L == 1L] <- "st"
  suffix[!teen & day %% 10L == 2L] <- "nd"
  suffix[!teen & day %% 10L == 3L] <- "rd"
  return(paste0(day, suffix))
}

#' Format dates in a named display style
#'
#' @param x A `Date` vector.
#' @param style One of `"iso"`, `"compact"`, `"long"`, `"short"`, `"us"`,
#'   `"ordinal"`, `"week"` or `"quarter"`.
#' @return A character vector the same length as `x`; missing dates give
#'   `NA`.
#' @export
format_date_style <- function(x, style = c("iso", "compact", "long", "short", "us",
                                           "ordinal", "week", "quarter")) {
  style <- match.arg(style)
  x <- as.Date(x)
  months <- c("January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December")
  lt <- as.POSIXlt(x)

  out <- switch(style,
    iso = {
      format(x, "%Y-%m-%d")
    },
    compact = {
      format(x, "%Y%m%d")
    },
    long = {
      paste(lt$mday, months[lt$mon + 1L], lt$year + 1900L)
    },
    short = {
      paste(lt$mday, substr(months[lt$mon + 1L], 1L, 3L), lt$year + 1900L)
    },
    us = {
      paste0(months[lt$mon + 1L], " ", lt$mday, ", ", lt$year + 1900L)
    },
    ordinal = {
      paste(ordinal_day(lt$mday), months[lt$mon + 1L], lt$year + 1900L)
    },
    week = {
      format(x, "%G-W%V")
    },
    quarter = {
      paste0(lt$year + 1900L, "-Q", lt$mon %/% 3L + 1L)
    }
  )
  out[is.na(x)] <- NA_character_
  return(out)
}

#' Format a `difftime` as a compact duration string
#'
#' @param x A `difftime` vector.
#' @param parts Maximum number of non-zero components to show.
#' @return A character vector such as `"2d 3h"` or `"45m 10s"`.
#' @export
format_duration <- function(x, parts = 2L) {
  secs <- round(abs(as.numeric(x, units = "secs")))
  sizes <- c(d = 86400, h = 3600, m = 60, s = 1)
  out <- character(length(secs))
  for (i in seq_along(secs)) {
    if (is.na(secs[i])) {
      out[i] <- NA_character_
      next
    }
    left <- secs[i]
    pieces <- character(0)
    for (lab in names(sizes)) {
      n <- left %/% sizes[[lab]]
      left <- left - n * sizes[[lab]]
      if (n > 0 && length(pieces) < parts) {
        pieces <- c(pieces, paste0(n, lab))
      }
    }
    if (length(pieces) == 0L) {
      pieces <- "0s"
    }
    sign <- if (as.numeric(x[i]) < 0) "-" else ""
    out[i] <- paste0(sign, paste(pieces, collapse = " "))
  }
  return(out)
}

#' Format a date range without repeating shared parts
#'
#' @param from,to Single `Date` values.
#' @param sep Separator placed between the two ends.
#' @return A single string such as `"3-9 March 2024"`,
#'   `"28 February - 3 March 2024"` or `"30 December 2024 - 2 January 2025"`.
#' @export
format_date_range <- function(from, to, sep = " - ") {
  from <- as.Date(from)
  to <- as.Date(to)
  if (from > to) {
    stop("`from` must not be after `to`.", call. = FALSE)
  }
  from_long <- format_date_style(from, "long")
  to_long <- format_date_style(to, "long")
  same_year <- format(from, "%Y") == format(to, "%Y")
  same_month <- same_year && format(from, "%m") == format(to, "%m")

  if (from == to) {
    return(from_long)
  }
  if (same_month) {
    from_day <- as.integer(format(from, "%d"))
    return(paste0(from_day, "-", to_long))
  }
  if (same_year) {
    from_part <- sub(" \\d{4}$", "", from_long)
    return(paste0(from_part, sep, to_long))
  }
  return(paste0(from_long, sep, to_long))
}

#' Format timestamps for a given resolution
#'
#' @param x A `POSIXct` vector.
#' @param resolution One of `"second"`, `"minute"`, `"hour"` or `"day"`.
#' @param tz Time zone used for display.
#' @return A character vector the same length as `x`.
#' @export
format_timestamp <- function(x, resolution = c("second", "minute", "hour", "day"),
                             tz = "UTC") {
  resolution <- match.arg(resolution)
  fmt <- switch(resolution,
    second = "%Y-%m-%d %H:%M:%S",
    minute = "%Y-%m-%d %H:%M",
    hour = "%Y-%m-%d %H:00",
    day = "%Y-%m-%d"
  )
  return(format(x, fmt, tz = tz))
}

#' Describe a date relative to a reference date
#'
#' @param x A `Date` vector.
#' @param today Reference date.
#' @return A character vector such as `"today"`, `"yesterday"`,
#'   `"in 3 days"` or `"2 weeks ago"`.
#' @export
relative_date_label <- function(x, today = Sys.Date()) {
  days <- as.integer(as.Date(x) - as.Date(today))
  n <- abs(days)
  unit <- ifelse(n >= 14L, "weeks", "days")
  count <- ifelse(n >= 14L, n %/% 7L, n)
  out <- ifelse(days > 0L, paste("in", count, unit), paste(count, unit, "ago"))
  out[days == 1L] <- "tomorrow"
  out[days == -1L] <- "yesterday"
  out[days == 0L] <- "today"
  out[is.na(days)] <- NA_character_
  return(out)
}
