#' Distance between two sets of points on a sphere
#'
#' @param lon1,lat1 Coordinates of the start points in degrees.
#' @param lon2,lat2 Coordinates of the end points in degrees.
#' @param metric One of `"haversine"`, `"cosine"`, `"equirect"`,
#'   `"vincenty"` or `"chord"`.
#' @param radius Sphere radius in metres.
#' @return A numeric vector of distances in metres.
#' @export
geo_distance <- function(lon1, lat1, lon2, lat2,
                         metric = c("haversine", "cosine", "equirect", "vincenty", "chord"),
                         radius = 6371008.8) {
  metric <- match.arg(metric)
  phi1 <- lat1 * pi / 180
  phi2 <- lat2 * pi / 180
  lam1 <- lon1 * pi / 180
  lam2 <- lon2 * pi / 180
  dphi <- phi2 - phi1
  dlam <- lam2 - lam1
  switch(metric,
    haversine = {
      a <- sin(dphi / 2)^2 + cos(phi1) * cos(phi2) * sin(dlam / 2)^2
      central <- 2 * asin(sqrt(pmin(a, 1)))
      radius * central
    },
    cosine = {
      a <- sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(dlam)
      central <- acos(pmin(pmax(a, -1), 1))
      radius * central
    },
    equirect = {
      # Fast approximation, fine for short legs away from the poles
      a <- dlam * cos((phi1 + phi2) / 2)
      central <- sqrt(a^2 + dphi^2)
      radius * central
    },
    vincenty = {
      num <- sqrt((cos(phi2) * sin(dlam))^2 +
        (cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlam))^2)
      den <- sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(dlam)
      central <- atan2(num, den)
      radius * central
    },
    chord = {
      dx <- cos(phi2) * cos(lam2) - cos(phi1) * cos(lam1)
      dy <- cos(phi2) * sin(lam2) - cos(phi1) * sin(lam1)
      dz <- sin(phi2) - sin(phi1)
      central <- sqrt(dx^2 + dy^2 + dz^2)
      radius * central
    }
  )
}

#' Bearing between two sets of points
#'
#' @param lon1,lat1 Coordinates of the start points in degrees.
#' @param lon2,lat2 Coordinates of the end points in degrees.
#' @param type One of `"initial"`, `"final"`, `"rhumb"` or `"reverse"`.
#' @return A numeric vector of bearings in degrees clockwise from north, in
#'   `[0, 360)`.
#' @export
geo_bearing <- function(lon1, lat1, lon2, lat2,
                        type = c("initial", "final", "rhumb", "reverse")) {
  type <- match.arg(type)
  phi1 <- lat1 * pi / 180
  phi2 <- lat2 * pi / 180
  dlam <- (lon2 - lon1) * pi / 180
  theta <- switch(type,
    initial = {
      y <- sin(dlam) * cos(phi2)
      x <- cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlam)
      atan2(y, x)
    },
    final = {
      # Initial bearing from the end point back, turned round
      y <- sin(-dlam) * cos(phi1)
      x <- cos(phi2) * sin(phi1) - sin(phi2) * cos(phi1) * cos(-dlam)
      atan2(y, x) + pi
    },
    rhumb = {
      dlam <- ifelse(abs(dlam) > pi, dlam - sign(dlam) * 2 * pi, dlam)
      y <- dlam
      x <- log(tan(pi / 4 + phi2 / 2) / tan(pi / 4 + phi1 / 2))
      atan2(y, x)
    },
    reverse = {
      y <- sin(-dlam) * cos(phi1)
      x <- cos(phi2) * sin(phi1) - sin(phi2) * cos(phi1) * cos(-dlam)
      atan2(y, x)
    }
  )
  (theta * 180 / pi) %% 360
}

#' Point reached from a start point, bearing and distance
#'
#' @param lon,lat Coordinates of the start points in degrees.
#' @param bearing Bearing in degrees clockwise from north.
#' @param distance Distance in metres.
#' @param method One of `"great_circle"`, `"rhumb"` or `"flat"`.
#' @param radius Sphere radius in metres.
#' @return A data frame with columns `lon` and `lat` in degrees.
#' @export
geo_destination <- function(lon, lat, bearing, distance,
                            method = c("great_circle", "rhumb", "flat"),
                            radius = 6371008.8) {
  method <- match.arg(method)
  phi1 <- lat * pi / 180
  lam1 <- lon * pi / 180
  theta <- bearing * pi / 180
  delta <- distance / radius
  switch(method,
    great_circle = {
      phi2 <- asin(sin(phi1) * cos(delta) + cos(phi1) * sin(delta) * cos(theta))
      lam2 <- lam1 + atan2(sin(theta) * sin(delta) * cos(phi1),
        cos(delta) - sin(phi1) * sin(phi2))
      lon2 <- ((lam2 * 180 / pi + 540) %% 360) - 180
      data.frame(lon = lon2, lat = phi2 * 180 / pi)
    },
    rhumb = {
      phi2 <- phi1 + delta * cos(theta)
      stretch <- log(tan(pi / 4 + phi2 / 2) / tan(pi / 4 + phi1 / 2))
      q <- ifelse(abs(stretch) > 1e-12, (phi2 - phi1) / stretch, cos(phi1))
      lam2 <- lam1 + delta * sin(theta) / q
      lon2 <- ((lam2 * 180 / pi + 540) %% 360) - 180
      data.frame(lon = lon2, lat = phi2 * 180 / pi)
    },
    flat = {
      phi2 <- phi1 + delta * cos(theta)
      lam2 <- lam1 + delta * sin(theta) / cos(phi1)
      lon2 <- ((lam2 * 180 / pi + 540) %% 360) - 180
      data.frame(lon = lon2, lat = phi2 * 180 / pi)
    }
  )
}

#' Leg statistics along a path
#'
#' @param lon,lat Coordinates of the path vertices in degrees, in order.
#' @param metric Passed to [geo_distance()].
#' @return A list with elements `legs` (distance of each leg), `cumulative`
#'   (distance from the first vertex), `total`, `longest`, `shortest` and
#'   `mean_leg`, all in metres.
#' @export
path_stats <- function(lon, lat, metric = "haversine") {
  n <- length(lon)
  if (n != length(lat)) {
    stop("`lon` and `lat` must have the same length.", call. = FALSE)
  }
  if (n < 2) {
    stop("A path needs at least two vertices.", call. = FALSE)
  }
  from <- seq_len(n - 1)
  to <- from + 1

  # 1. legs
  legs <- geo_distance(lon[from], lat[from], lon[to], lat[to], metric = metric)

  # 2. cumulative
  cumulative <- c(0, cumsum(legs))

  # 3. total, longest, shortest, mean_leg
  total <- sum(legs)
  longest <- max(legs)
  shortest <- min(legs)
  mean_leg <- total / length(legs)
  list(
    legs = legs,
    cumulative = cumulative,
    total = total,
    longest = longest,
    shortest = shortest,
    mean_leg = mean_leg
  )
}

#' Convert distances between units
#'
#' @param x Numeric vector of distances.
#' @param from,to Units: `"m"`, `"km"`, `"mi"`, `"nmi"`, `"ft"` or `"yd"`.
#' @return A numeric vector of distances in `to` units, with attribute
#'   `units` set to `to`.
#' @keywords internal
convert_distance <- function(x, from = c("m", "km", "mi", "nmi", "ft", "yd"),
                             to = c("m", "km", "mi", "nmi", "ft", "yd")) {
  from <- match.arg(from)
  to <- match.arg(to)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }

  # Step 1: from `from` to metres
  metres <- switch(from,
    m = x,
    km = x * 1000,
    mi = x * 1609.344,
    nmi = x * 1852,
    ft = x * 0.3048,
    yd = x * 0.9144
  )

  # Step 2: from metres to `to`
  out <- switch(to,
    m = metres,
    km = metres / 1000,
    mi = metres / 1609.344,
    nmi = metres / 1852,
    ft = metres / 0.3048,
    yd = metres / 0.9144
  )
  attr(out, "units") <- to
  out
}

#' Nearest reference point for each query point
#'
#' @param lon,lat Coordinates of the query points in degrees.
#' @param ref_lon,ref_lat Coordinates of the reference points in degrees.
#' @param metric Passed to [geo_distance()].
#' @param max_dist Distances above this many metres count as no match.
#' @return A data frame with columns `index` (row of the nearest reference
#'   point, `NA` if none) and `distance` in metres.
#' @export
nearest_point <- function(lon, lat, ref_lon, ref_lat, metric = "haversine",
                          max_dist = Inf) {
  n <- length(lon)
  index <- rep(NA_integer_, n)
  distance <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    if (is.na(lon[i]) || is.na(lat[i])) {
      next
    }
    d <- geo_distance(lon[i], lat[i], ref_lon, ref_lat, metric = metric)
    best <- which.min(d)
    if (length(best) == 1 && d[best] <= max_dist) {
      index[i] <- best
      distance[i] <- d[best]
    }
  }
  data.frame(index = index, distance = distance)
}
