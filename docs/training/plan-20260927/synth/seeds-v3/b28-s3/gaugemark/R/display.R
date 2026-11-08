#' @title Format readings
#' @description Each reading is rendered as `value unit`; missing values as
#'   `NA`. `notation` picks how `value` is written: `"plain"` rounds to
#'   `digits` significant digits, `"fixed"` uses `digits` decimals, and
#'   `"scientific"` uses exponent notation.
#' @param x An object of class `gm_reading`.
#' @param digits Number of digits, interpreted per `notation`.
#' @param notation One of `"plain"`, `"fixed"` or `"scientific"`.
#' @param ... Unused.
#' @return A character vector with one element per reading.
#' @export
format.gm_reading = function(x, digits = 3, notation = c("plain", "fixed", "scientific"),
                             ...) {
  notation = match.arg(notation)
  unit = attr(x, "unit")
  value = as.double(unclass(x))
  text = switch(notation,
    plain = {
      as.character(signif(value, digits))
    },
    fixed = {
      formatC(value, digits = digits, format = "f")
    },
    scientific = {
      formatC(value, digits = digits, format = "e")
    }
  )
  out = paste(trimws(text), unit)
  out[is.na(value)] = "NA"
  out
}

#' @title Format ranges
#' @description Each range is rendered as `lower-upper unit`; ranges with a
#'   missing bound as `NA`. `notation` works as in [format.gm_reading()] and
#'   applies to both bounds.
#' @param x An object of class `gm_range`.
#' @param digits Number of digits, interpreted per `notation`.
#' @param notation One of `"plain"`, `"fixed"` or `"scientific"`.
#' @param ... Unused.
#' @return A character vector with one element per range.
#' @export
format.gm_range = function(x, digits = 3, notation = c("plain", "fixed", "scientific"),
                           ...) {
  notation = match.arg(notation)
  unit = attr(x, "unit")
  bounds = unclass(x)
  switch(notation,
    plain = {
      lower = as.character(signif(bounds$lower, digits))
      upper = as.character(signif(bounds$upper, digits))
    },
    fixed = {
      lower = formatC(bounds$lower, digits = digits, format = "f")
      upper = formatC(bounds$upper, digits = digits, format = "f")
    },
    scientific = {
      lower = formatC(bounds$lower, digits = digits, format = "e")
      upper = formatC(bounds$upper, digits = digits, format = "e")
    }
  )
  out = paste0(trimws(lower), "-", trimws(upper), " ", unit)
  out[is.na(bounds$lower) | is.na(bounds$upper)] = "NA"
  out
}

#' @title Print readings
#' @param x An object of class `gm_reading`.
#' @param ... Passed to [format.gm_reading()].
#' @return `x`, invisibly.
#' @export
print.gm_reading = function(x, ...) {
  n = length(unclass(x))
  cat("<gm_reading[", n, "]>\n", sep = "")
  if (n > 0L) {
    print(format(x, ...), quote = FALSE)
  }
  invisible(x)
}

#' @title Print ranges
#' @param x An object of class `gm_range`.
#' @param ... Passed to [format.gm_range()].
#' @return `x`, invisibly.
#' @export
print.gm_range = function(x, ...) {
  n = length(unclass(x)$lower)
  cat("<gm_range[", n, "]>\n", sep = "")
  if (n > 0L) {
    print(format(x, ...), quote = FALSE)
  }
  invisible(x)
}

#' @title Number of ranges
#' @param x An object of class `gm_range`.
#' @return A single integer.
#' @export
length.gm_range = function(x) {
  length(unclass(x)$lower)
}

#' @title Subset readings
#' @description Keeps the unit; names are dropped.
#' @param x An object of class `gm_reading`.
#' @param i Positions or a logical vector.
#' @return An object of class `gm_reading`.
#' @export
`[.gm_reading` = function(x, i) {
  unit = attr(x, "unit")
  value = as.double(unclass(x))
  if (missing(i)) {
    return(x)
  }
  newReading(value[i], unit)
}

#' @title Subset ranges
#' @description Keeps the unit.
#' @param x An object of class `gm_range`.
#' @param i Positions or a logical vector.
#' @return An object of class `gm_range`.
#' @export
`[.gm_range` = function(x, i) {
  unit = attr(x, "unit")
  bounds = unclass(x)
  if (missing(i)) {
    return(x)
  }
  newRange(bounds$lower[i], bounds$upper[i], unit)
}
