#' @title Convert angles between units
#' @param x A numeric vector of angles.
#' @param from,to Units: `"deg"`, `"rad"`, `"grad"`, `"arcmin"`, `"arcsec"`,
#'   `"turn"`, `"mil"` (6400 per turn) or `"hour"` (24 per turn).
#' @return A numeric vector of angles in `to` units.
#' @export
angle.convert <- function(x,
                          from = c("deg", "rad", "grad", "arcmin", "arcsec", "turn", "mil",
                                   "hour"),
                          to = c("deg", "rad", "grad", "arcmin", "arcsec", "turn", "mil",
                                 "hour")) {
  from <- match.arg(from)
  to <- match.arg(to)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }

  # Step 1: `from` units to degrees
  deg <- switch(from,
    deg = x,
    rad = x * 180 / pi,
    grad = x * 360 / 400,
    arcmin = x / 60,
    arcsec = x / 3600,
    turn = x * 360,
    mil = x * 360 / 6400,
    hour = x * 360 / 24
  )

  # Step 2: degrees to `to` units
  switch(to,
    deg = deg,
    rad = deg * pi / 180,
    grad = deg * 400 / 360,
    arcmin = deg * 60,
    arcsec = deg * 3600,
    turn = deg / 360,
    mil = deg * 6400 / 360,
    hour = deg * 24 / 360
  )
}

#' @title Wrap angles into a standard range
#' @param x A numeric vector of angles.
#' @param range One of `"0_360"` and `"pm180"` (degrees), `"0_2pi"` and
#'   `"pm_pi"` (radians), or `"0_400"` and `"pm200"` (grads).
#' @return A numeric vector of wrapped angles.
#' @export
angle.normalize <- function(x, range = c("0_360", "pm180", "0_2pi", "pm_pi", "0_400",
                                         "pm200")) {
  range <- match.arg(range)
  switch(range,
    "0_360" = {
      period <- 360
      out <- x %% period
      out
    },
    pm180 = {
      period <- 360
      out <- ((x + period / 2) %% period) - period / 2
      out
    },
    "0_2pi" = {
      period <- 2 * pi
      out <- x %% period
      out
    },
    pm_pi = {
      period <- 2 * pi
      out <- ((x + period / 2) %% period) - period / 2
      out
    },
    "0_400" = {
      period <- 400
      out <- x %% period
      out
    },
    pm200 = {
      period <- 400
      out <- ((x + period / 2) %% period) - period / 2
      out
    }
  )
}

#' @title Reference ellipsoid parameters
#' @param name One of `"wgs84"`, `"grs80"`, `"clarke1866"`, `"intl1924"`,
#'   `"airy1830"` or `"sphere"`.
#' @return A list with elements `name`, `a` (semi-major axis, metres), `f`
#'   (flattening), `b` (semi-minor axis), `e2` (first eccentricity squared)
#'   and `ep2` (second eccentricity squared).
#' @keywords internal
ellipsoid.params <- function(name = c("wgs84", "grs80", "clarke1866", "intl1924", "airy1830",
                                     "sphere")) {
  name <- match.arg(name)
  axes <- switch(name,
    wgs84 = {
      a <- 6378137
      inv.f <- 298.257223563
      c(a = a, f = 1 / inv.f)
    },
    grs80 = {
      a <- 6378137
      inv.f <- 298.257222101
      c(a = a, f = 1 / inv.f)
    },
    clarke1866 = {
      a <- 6378206.4
      inv.f <- 294.978698214
      c(a = a, f = 1 / inv.f)
    },
    intl1924 = {
      a <- 6378388
      inv.f <- 297
      c(a = a, f = 1 / inv.f)
    },
    airy1830 = {
      a <- 6377563.396
      inv.f <- 299.3249646
      c(a = a, f = 1 / inv.f)
    },
    sphere = {
      a <- 6371008.8
      inv.f <- Inf
      c(a = a, f = 1 / inv.f)
    }
  )
  a <- axes[["a"]]
  f <- axes[["f"]]
  b <- a * (1 - f)
  e2 <- f * (2 - f)
  ep2 <- e2 / (1 - e2)
  list(name = name, a = a, f = f, b = b, e2 = e2, ep2 = ep2)
}

#' @title Geodetic to earth-centred cartesian coordinates
#' @param lon,lat Coordinates in degrees.
#' @param h Height above the ellipsoid in metres.
#' @param ellipsoid Passed to [ellipsoid.params()].
#' @return A data frame with columns `x`, `y` and `z` in metres.
#' @export
geodetic.to.ecef <- function(lon, lat, h = 0, ellipsoid = "wgs84") {
  ell <- ellipsoid.params(ellipsoid)
  lam <- lon * pi / 180
  phi <- lat * pi / 180

  # Prime vertical radius of curvature
  n <- ell$a / sqrt(1 - ell$e2 * sin(phi)^2)

  # Cartesian coordinates
  x <- (n + h) * cos(phi) * cos(lam)
  y <- (n + h) * cos(phi) * sin(lam)
  z <- (n * (1 - ell$e2) + h) * sin(phi)
  data.frame(x = x, y = y, z = z)
}

#' @title Earth-centred cartesian to geodetic coordinates
#' @param x,y,z Cartesian coordinates in metres.
#' @param ellipsoid Passed to [ellipsoid.params()].
#' @return A data frame with columns `lon`, `lat` (degrees) and `h`
#'   (metres).
#' @export
ecef.to.geodetic <- function(x, y, z, ellipsoid = "wgs84") {
  ell <- ellipsoid.params(ellipsoid)
  p <- sqrt(x^2 + y^2)

  # Bowring's parametric latitude, one step
  theta <- atan2(z * ell$a, p * ell$b)
  phi <- atan2(z + ell$ep2 * ell$b * sin(theta)^3, p - ell$e2 * ell$a * cos(theta)^3)

  # Prime vertical radius of curvature
  n <- ell$a / sqrt(1 - ell$e2 * sin(phi)^2)

  # Geodetic coordinates
  lam <- atan2(y, x)
  h <- ifelse(abs(cos(phi)) > 1e-10, p / cos(phi) - n, abs(z) - ell$b)
  data.frame(lon = lam * 180 / pi, lat = phi * 180 / pi, h = h)
}

#' @title Split angles into whole units and remainders
#' @param x A numeric vector of decimal degrees (or hours for `"hms"`).
#' @param unit One of `"dms"`, `"dm"`, `"hms"` or `"hm"`.
#' @return A data frame with columns `sign`, `whole`, `minutes` and
#'   `seconds`; `seconds` is `0` and `minutes` keeps its fraction for `"dm"`
#'   and `"hm"`.
#' @export
angle.parts <- function(x, unit = c("dms", "dm", "hms", "hm")) {
  unit <- match.arg(unit)
  sign <- ifelse(x < 0, -1, 1)
  a <- abs(x)
  switch(unit,
    dms = {
      whole <- floor(a)
      minutes <- floor((a - whole) * 60)
      seconds <- ((a - whole) * 60 - minutes) * 60
      data.frame(sign = sign, whole = whole, minutes = minutes, seconds = seconds)
    },
    dm = {
      whole <- floor(a)
      minutes <- (a - whole) * 60
      seconds <- 0
      data.frame(sign = sign, whole = whole, minutes = minutes, seconds = seconds)
    },
    hms = {
      a <- a %% 24
      whole <- floor(a)
      minutes <- floor((a - whole) * 60)
      seconds <- ((a - whole) * 60 - minutes) * 60
      data.frame(sign = sign, whole = whole, minutes = minutes, seconds = seconds)
    },
    hm = {
      a <- a %% 24
      whole <- floor(a)
      minutes <- (a - whole) * 60
      seconds <- 0
      data.frame(sign = sign, whole = whole, minutes = minutes, seconds = seconds)
    }
  )
}
