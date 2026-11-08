#' Validate coordinate vectors axis by axis
#'
#' Checks `lon`, `lat` and, when given, `alt` for type, missingness and range.
#'
#' @param lon Numeric vector of longitudes in degrees.
#' @param lat Numeric vector of latitudes in degrees.
#' @param alt Optional numeric vector of altitudes in metres.
#' @param na_ok Whether missing values are allowed.
#' @return A list with elements `lon`, `lat` and `alt`, invisibly.
#' @export
validate_coords <- function(lon, lat, alt = NULL, na_ok = FALSE) {
  limits <- list(lon = c(-180, 180), lat = c(-90, 90), alt = c(-11000, 9000))
  n <- length(lon)

  # Validate `lon`
  if (!is.numeric(lon)) {
    stop(sprintf("`%s` must be numeric.", "lon"), call. = FALSE)
  }
  if (!na_ok && anyNA(lon)) {
    stop(sprintf("`%s` must not contain missing values.", "lon"), call. = FALSE)
  }
  if (any(lon < limits$lon[1] | lon > limits$lon[2], na.rm = TRUE)) {
    stop(sprintf("`%s` must lie in [%g, %g].", "lon", limits$lon[1], limits$lon[2]),
      call. = FALSE)
  }

  # Validate `lat`
  if (!is.numeric(lat)) {
    stop(sprintf("`%s` must be numeric.", "lat"), call. = FALSE)
  }
  if (!na_ok && anyNA(lat)) {
    stop(sprintf("`%s` must not contain missing values.", "lat"), call. = FALSE)
  }
  if (any(lat < limits$lat[1] | lat > limits$lat[2], na.rm = TRUE)) {
    stop(sprintf("`%s` must lie in [%g, %g].", "lat", limits$lat[1], limits$lat[2]),
      call. = FALSE)
  }
  if (length(lat) != n) {
    stop(sprintf("`%s` must have the same length as `lon`.", "lat"), call. = FALSE)
  }

  # Validate `alt`
  if (!is.null(alt)) {
    if (!is.numeric(alt)) {
      stop(sprintf("`%s` must be numeric.", "alt"), call. = FALSE)
    }
    if (!na_ok && anyNA(alt)) {
      stop(sprintf("`%s` must not contain missing values.", "alt"), call. = FALSE)
    }
    if (any(alt < limits$alt[1] | alt > limits$alt[2], na.rm = TRUE)) {
      stop(sprintf("`%s` must lie in [%g, %g].", "alt", limits$alt[1], limits$alt[2]),
        call. = FALSE)
    }
    if (length(alt) != n) {
      stop(sprintf("`%s` must have the same length as `lon`.", "alt"), call. = FALSE)
    }
  }
  invisible(list(lon = lon, lat = lat, alt = alt))
}

#' Bring out-of-range coordinates back into range
#'
#' @param lon Numeric vector of longitudes in degrees.
#' @param lat Numeric vector of latitudes in degrees.
#' @param mode How to treat values outside the valid range: `"clamp"`,
#'   `"wrap"`, `"reflect"`, `"drop"` or `"na"`.
#' @return A data frame with columns `lon` and `lat`.
#' @export
normalize_coords <- function(lon, lat, mode = c("clamp", "wrap", "reflect", "drop", "na")) {
  mode <- match.arg(mode)
  stopifnot(is.numeric(lon), is.numeric(lat), length(lon) == length(lat))
  out <- switch(mode,
    clamp = {
      lon <- pmin(pmax(lon, -180), 180)
      lat <- pmin(pmax(lat, -90), 90)
      data.frame(lon = lon, lat = lat)
    },
    wrap = {
      lon <- ((lon + 180) %% 360) - 180
      lat <- ((lat + 90) %% 180) - 90
      data.frame(lon = lon, lat = lat)
    },
    reflect = {
      # Crossing a pole flips the latitude and moves half way round
      over <- abs(lat) > 90
      lat[over] <- sign(lat[over]) * 180 - lat[over]
      lon[over] <- lon[over] + 180
      lon <- ((lon + 180) %% 360) - 180
      data.frame(lon = lon, lat = lat)
    },
    drop = {
      keep <- abs(lon) <= 180 & abs(lat) <= 90
      lon <- lon[keep]
      lat <- lat[keep]
      data.frame(lon = lon, lat = lat)
    },
    na = {
      bad <- abs(lon) > 180 | abs(lat) > 90
      lon[bad] <- NA
      lat[bad] <- NA
      data.frame(lon = lon, lat = lat)
    }
  )
  rownames(out) <- NULL
  out
}

#' Summarise each coordinate column
#'
#' @param coords A data frame with columns `lon`, `lat` and `alt`.
#' @return A data frame with one row per axis and columns `axis`, `n`,
#'   `n_missing`, `min`, `max`, `mean` and `span`.
#' @keywords internal
summarise_coords <- function(coords) {
  axes <- c("lon", "lat", "alt")
  missing_cols <- setdiff(axes, names(coords))
  if (length(missing_cols) > 0) {
    stop(sprintf("`coords` is missing column(s): %s.",
      paste(missing_cols, collapse = ", ")), call. = FALSE)
  }
  rows <- vector("list", length(axes))

  # Summarise `lon`
  x <- coords$lon
  ok <- !is.na(x)
  rows[[1]] <- data.frame(axis = "lon", n = length(x), n_missing = sum(!ok),
    min = min(x[ok]), max = max(x[ok]), mean = mean(x[ok]))

  # Summarise `lat`
  x <- coords$lat
  ok <- !is.na(x)
  rows[[2]] <- data.frame(axis = "lat", n = length(x), n_missing = sum(!ok),
    min = min(x[ok]), max = max(x[ok]), mean = mean(x[ok]))

  # Summarise `alt`
  x <- coords$alt
  ok <- !is.na(x)
  rows[[3]] <- data.frame(axis = "alt", n = length(x), n_missing = sum(!ok),
    min = min(x[ok]), max = max(x[ok]), mean = mean(x[ok]))

  out <- do.call(rbind, rows)
  out$span <- out$max - out$min
  out
}

#' Flag suspicious coordinates
#'
#' Each flag is a logical column. A point can carry several flags.
#'
#' @param lon Numeric vector of longitudes in degrees.
#' @param lat Numeric vector of latitudes in degrees.
#' @param digits Minimum number of decimal places expected.
#' @return A data frame with columns `lon`, `lat`, `missing`,
#'   `out_of_range`, `null_island`, `swapped`, `low_precision`, `pole`,
#'   `duplicate` and `any`.
#' @export
flag_coords <- function(lon, lat, digits = 2) {
  stopifnot(length(lon) == length(lat))
  out <- data.frame(lon = lon, lat = lat)
  decimals <- function(v) {
    s <- format(abs(v), scientific = FALSE, drop0trailing = TRUE, trim = TRUE)
    ifelse(grepl(".", s, fixed = TRUE), nchar(sub("^[^.]*\\.", "", s)), 0L)
  }

  # Flag `missing`
  out$missing <- is.na(lon) | is.na(lat)

  # Flag `out_of_range`
  out$out_of_range <- !out$missing & (abs(lon) > 180 | abs(lat) > 90)

  # Flag `null_island`
  out$null_island <- !out$missing & lon == 0 & lat == 0

  # Flag `swapped`
  out$swapped <- !out$missing & abs(lat) > 90 & abs(lon) <= 90

  # Flag `low_precision`
  out$low_precision <- !out$missing &
    (decimals(lon) < digits | decimals(lat) < digits)

  # Flag `pole`
  out$pole <- !out$missing & abs(lat) == 90

  # Flag `duplicate`
  out$duplicate <- !out$missing & duplicated(data.frame(lon, lat))

  flags <- c("missing", "out_of_range", "null_island", "swapped", "low_precision", "pole",
    "duplicate")
  out$any <- Reduce(`|`, out[flags])
  out
}

#' Format coordinates as text
#'
#' @param lon Numeric vector of longitudes in degrees.
#' @param lat Numeric vector of latitudes in degrees.
#' @param style One of `"decimal"`, `"dms"`, `"dm"` or `"hemisphere"`.
#' @param digits Number of decimal places on the last component.
#' @return A character vector of `"lat, lon"` labels.
#' @export
format_coords <- function(lon, lat, style = c("decimal", "dms", "dm", "hemisphere"),
                          digits = 4) {
  style <- match.arg(style)
  hemi_lat <- ifelse(lat < 0, "S", "N")
  hemi_lon <- ifelse(lon < 0, "W", "E")
  switch(style,
    decimal = {
      lat_txt <- formatC(lat, format = "f", digits = digits)
      lon_txt <- formatC(lon, format = "f", digits = digits)
      paste0(lat_txt, ", ", lon_txt)
    },
    dms = {
      d_lat <- floor(abs(lat))
      m_lat <- floor((abs(lat) - d_lat) * 60)
      s_lat <- ((abs(lat) - d_lat) * 60 - m_lat) * 60
      d_lon <- floor(abs(lon))
      m_lon <- floor((abs(lon) - d_lon) * 60)
      s_lon <- ((abs(lon) - d_lon) * 60 - m_lon) * 60
      lat_txt <- sprintf("%d %d' %.*f\" %s", d_lat, m_lat, digits, s_lat, hemi_lat)
      lon_txt <- sprintf("%d %d' %.*f\" %s", d_lon, m_lon, digits, s_lon, hemi_lon)
      paste0(lat_txt, ", ", lon_txt)
    },
    dm = {
      d_lat <- floor(abs(lat))
      m_lat <- (abs(lat) - d_lat) * 60
      d_lon <- floor(abs(lon))
      m_lon <- (abs(lon) - d_lon) * 60
      lat_txt <- sprintf("%d %.*f' %s", d_lat, digits, m_lat, hemi_lat)
      lon_txt <- sprintf("%d %.*f' %s", d_lon, digits, m_lon, hemi_lon)
      paste0(lat_txt, ", ", lon_txt)
    },
    hemisphere = {
      lat_txt <- paste(formatC(abs(lat), format = "f", digits = digits), hemi_lat)
      lon_txt <- paste(formatC(abs(lon), format = "f", digits = digits), hemi_lon)
      paste0(lat_txt, ", ", lon_txt)
    }
  )
}

#' Coerce assorted inputs to a coordinate data frame
#'
#' @param x A two-column matrix, a list with `lon` and `lat`, a character
#'   vector of `"lon lat"` pairs, a data frame, or a character vector of
#'   WKT points such as `"POINT (13.4 52.5)"`.
#' @param from The input layout: `"matrix"`, `"list"`, `"character"`,
#'   `"data.frame"` or `"wkt"`.
#' @return A data frame with numeric columns `lon` and `lat`.
#' @keywords internal
as_coords <- function(x, from = c("matrix", "list", "character", "data.frame", "wkt")) {
  from <- match.arg(from)
  out <- switch(from,
    matrix = {
      if (!is.matrix(x) || ncol(x) != 2) {
        stop(sprintf("`x` must be a two-column %s.", "matrix"), call. = FALSE)
      }
      data.frame(lon = as.numeric(x[, 1]), lat = as.numeric(x[, 2]))
    },
    list = {
      if (!is.list(x) || !all(c("lon", "lat") %in% names(x))) {
        stop(sprintf("`x` must be a two-column %s.", "list"), call. = FALSE)
      }
      data.frame(lon = as.numeric(x$lon), lat = as.numeric(x$lat))
    },
    character = {
      parts <- strsplit(trimws(x), "[[:space:],;]+")
      if (!is.character(x) || any(lengths(parts) != 2)) {
        stop(sprintf("`x` must be a two-column %s.", "character"), call. = FALSE)
      }
      data.frame(lon = as.numeric(vapply(parts, `[`, "", 1)),
        lat = as.numeric(vapply(parts, `[`, "", 2)))
    },
    data.frame = {
      if (!is.data.frame(x) || !all(c("lon", "lat") %in% names(x))) {
        stop(sprintf("`x` must be a two-column %s.", "data.frame"), call. = FALSE)
      }
      data.frame(lon = as.numeric(x$lon), lat = as.numeric(x$lat))
    },
    wkt = {
      pattern <- "^\\s*POINT\\s*\\(\\s*(\\S+)\\s+(\\S+)\\s*\\)\\s*$"
      if (!is.character(x) || !all(grepl(pattern, x, ignore.case = TRUE))) {
        stop(sprintf("`x` must be a two-column %s.", "wkt"), call. = FALSE)
      }
      data.frame(lon = as.numeric(sub(pattern, "\\1", x, ignore.case = TRUE)),
        lat = as.numeric(sub(pattern, "\\2", x, ignore.case = TRUE)))
    }
  )
  validate_coords(out$lon, out$lat, na_ok = TRUE)
  out
}
