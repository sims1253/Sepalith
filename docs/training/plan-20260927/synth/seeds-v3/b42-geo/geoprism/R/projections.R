#' Project geographic coordinates to planar coordinates
#'
#' Spherical forward formulas for a few cylindrical and pseudo-cylindrical
#' projections centred on the prime meridian.
#'
#' @param lon Numeric vector of longitudes in degrees.
#' @param lat Numeric vector of latitudes in degrees.
#' @param projection One of `"equirectangular"`, `"mercator"`,
#'   `"web_mercator"`, `"sinusoidal"`, `"lambert_cea"` or `"miller"`.
#' @param radius Sphere radius in metres.
#' @return A data frame with columns `x` and `y` in metres.
#' @export
project_forward <- function(lon, lat,
                            projection = c("equirectangular", "mercator", "web_mercator",
                                           "sinusoidal", "lambert_cea", "miller"),
                            radius = 6378137) {
  projection <- match.arg(projection)
  stopifnot(length(lon) == length(lat))
  lam <- lon * pi / 180
  phi <- lat * pi / 180
  switch(projection,
    equirectangular = {
      x <- radius * lam
      y <- radius * phi
      data.frame(x = x, y = y)
    },
    mercator = {
      if (any(abs(lat) >= 90, na.rm = TRUE)) {
        stop(sprintf("`%s` cannot show the poles.", projection), call. = FALSE)
      }
      x <- radius * lam
      y <- radius * log(tan(pi / 4 + phi / 2))
      data.frame(x = x, y = y)
    },
    web_mercator = {
      # Latitudes beyond the square world limit are clamped, not rejected
      max_lat <- atan(sinh(pi)) * 180 / pi
      phi <- pmin(pmax(lat, -max_lat), max_lat) * pi / 180
      x <- radius * lam
      y <- radius * log(tan(pi / 4 + phi / 2))
      data.frame(x = x, y = y)
    },
    sinusoidal = {
      x <- radius * lam * cos(phi)
      y <- radius * phi
      data.frame(x = x, y = y)
    },
    lambert_cea = {
      x <- radius * lam
      y <- radius * sin(phi)
      data.frame(x = x, y = y)
    },
    miller = {
      x <- radius * lam
      y <- radius * 1.25 * log(tan(pi / 4 + 0.4 * phi))
      data.frame(x = x, y = y)
    }
  )
}

#' Unproject planar coordinates back to geographic coordinates
#'
#' Inverse of [project_forward()] for the same set of projections.
#'
#' @inheritParams project_forward
#' @param x,y Numeric vectors of planar coordinates in metres.
#' @return A data frame with columns `lon` and `lat` in degrees.
#' @export
project_inverse <- function(x, y,
                            projection = c("equirectangular", "mercator", "web_mercator",
                                           "sinusoidal", "lambert_cea", "miller"),
                            radius = 6378137) {
  projection <- match.arg(projection)
  stopifnot(length(x) == length(y))
  to_deg <- 180 / pi
  switch(projection,
    equirectangular = {
      lam <- x / radius
      phi <- y / radius
      data.frame(lon = lam * to_deg, lat = phi * to_deg)
    },
    mercator = {
      lam <- x / radius
      phi <- 2 * atan(exp(y / radius)) - pi / 2
      data.frame(lon = lam * to_deg, lat = phi * to_deg)
    },
    web_mercator = {
      lam <- x / radius
      phi <- 2 * atan(exp(y / radius)) - pi / 2
      data.frame(lon = lam * to_deg, lat = phi * to_deg)
    },
    sinusoidal = {
      phi <- y / radius
      lam <- ifelse(abs(cos(phi)) < 1e-12, 0, x / (radius * cos(phi)))
      data.frame(lon = lam * to_deg, lat = phi * to_deg)
    },
    lambert_cea = {
      lam <- x / radius
      phi <- asin(pmin(pmax(y / radius, -1), 1))
      data.frame(lon = lam * to_deg, lat = phi * to_deg)
    },
    miller = {
      lam <- x / radius
      phi <- 2.5 * atan(exp(0.8 * y / radius)) - 0.625 * pi
      data.frame(lon = lam * to_deg, lat = phi * to_deg)
    }
  )
}

#' Projected extent of the whole world
#'
#' @inheritParams project_forward
#' @param max_lat Latitude limit used for projections that cannot show the
#'   poles.
#' @return A named numeric vector with elements `xmin`, `ymin`, `xmax` and
#'   `ymax` in metres.
#' @keywords internal
projection_extent <- function(projection = c("equirectangular", "mercator", "web_mercator",
                                             "sinusoidal", "lambert_cea", "miller"),
                              radius = 6378137, max_lat = 85) {
  projection <- match.arg(projection)
  half_x <- pi * radius
  half_y <- switch(projection,
    equirectangular = {
      radius * pi / 2
    },
    mercator = {
      phi <- max_lat * pi / 180
      radius * log(tan(pi / 4 + phi / 2))
    },
    web_mercator = {
      half_x
    },
    sinusoidal = {
      radius * pi / 2
    },
    lambert_cea = {
      radius
    },
    miller = {
      radius * 1.25 * log(tan(pi / 4 + 0.4 * pi / 2))
    }
  )
  c(xmin = -half_x, ymin = -half_y, xmax = half_x, ymax = half_y)
}

#' Scale distortion along meridians and parallels
#'
#' @param lat Numeric vector of latitudes in degrees.
#' @param projection One of `"equirectangular"`, `"mercator"`,
#'   `"web_mercator"`, `"lambert_cea"` or `"miller"`.
#' @return A data frame with columns `lat`, `h` (meridian scale), `k`
#'   (parallel scale), `area` (`h * k`) and `angle` (maximum angular
#'   distortion in degrees).
#' @export
projection_distortion <- function(lat,
                                  projection = c("equirectangular", "mercator",
                                                 "web_mercator", "lambert_cea",
                                                 "miller")) {
  projection <- match.arg(projection)
  if (any(abs(lat) >= 90, na.rm = TRUE)) {
    stop("`lat` must be strictly between -90 and 90.", call. = FALSE)
  }
  phi <- lat * pi / 180
  sec <- 1 / cos(phi)
  scales <- switch(projection,
    equirectangular = {
      h <- rep(1, length(phi))
      k <- sec
      list(h = h, k = k)
    },
    mercator = {
      h <- sec
      k <- sec
      list(h = h, k = k)
    },
    web_mercator = {
      h <- sec
      k <- sec
      list(h = h, k = k)
    },
    lambert_cea = {
      h <- cos(phi)
      k <- sec
      list(h = h, k = k)
    },
    miller = {
      h <- 1 / cos(0.8 * phi)
      k <- sec
      list(h = h, k = k)
    }
  )
  h <- scales$h
  k <- scales$k

  # Tissot: sin(omega / 2) = |h - k| / (h + k)
  angle <- 2 * asin(abs(h - k) / (h + k)) * 180 / pi
  data.frame(lat = lat, h = h, k = k, area = h * k, angle = angle)
}

#' Reproject the coordinate columns of a data frame
#'
#' @param data A data frame.
#' @param from,to Source and target projections, as accepted by
#'   [project_forward()], or `"lonlat"` for geographic degrees.
#' @param cols Names of the two coordinate columns, x first.
#' @param radius Sphere radius in metres.
#' @return `data` with the coordinate columns replaced and the attribute
#'   `projection` set to `to`.
#' @export
reproject_df <- function(data, from, to, cols = c("x", "y"), radius = 6378137) {
  choices <- c("lonlat", "equirectangular", "mercator", "web_mercator",
    "sinusoidal", "lambert_cea", "miller")
  from <- match.arg(from, choices)
  to <- match.arg(to, choices)
  if (!all(cols %in% names(data))) {
    stop(sprintf("`data` must contain columns %s.", paste(cols, collapse = " and ")),
      call. = FALSE)
  }
  a <- data[[cols[1]]]
  b <- data[[cols[2]]]

  # Step 1: source coordinates to degrees
  if (from == "lonlat") {
    geo <- data.frame(lon = a, lat = b)
  } else {
    geo <- project_inverse(a, b, from, radius = radius)
  }

  # Step 2: degrees to target coordinates
  if (to == "lonlat") {
    res <- data.frame(x = geo$lon, y = geo$lat)
  } else {
    res <- project_forward(geo$lon, geo$lat, to, radius = radius)
  }
  data[[cols[1]]] <- res$x
  data[[cols[2]]] <- res$y
  attr(data, "projection") <- to
  data
}
