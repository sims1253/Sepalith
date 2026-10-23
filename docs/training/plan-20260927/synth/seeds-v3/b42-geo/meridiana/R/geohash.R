#' @title Encode points as geohashes
#' @description Bits alternate between longitude (even bits) and latitude
#'   (odd bits); every five bits make one base-32 character.
#' @param lon,lat Coordinates in degrees.
#' @param precision Number of characters per hash.
#' @return A character vector of geohashes.
#' @export
geohash.encode <- function(lon, lat, precision = 9L) {
  alphabet <- strsplit("0123456789bcdefghjkmnpqrstuvwxyz", "")[[1]]
  stopifnot(length(lon) == length(lat), precision >= 1, precision <= 12)
  n.bits <- 5L * precision
  vapply(seq_along(lon), function(i) {
    lon.lo <- -180
    lon.hi <- 180
    lat.lo <- -90
    lat.hi <- 90
    bits <- integer(n.bits)
    for (b in seq_len(n.bits)) {
      if (b %% 2 == 1) {
        # Halve the longitude interval
        mid <- (lon.lo + lon.hi) / 2
        bits[b] <- as.integer(lon[i] >= mid)
        if (bits[b] == 1L) lon.lo <- mid else lon.hi <- mid
      } else {
        # Halve the latitude interval
        mid <- (lat.lo + lat.hi) / 2
        bits[b] <- as.integer(lat[i] >= mid)
        if (bits[b] == 1L) lat.lo <- mid else lat.hi <- mid
      }
    }
    groups <- matrix(bits, nrow = 5L)
    paste(alphabet[colSums(groups * c(16L, 8L, 4L, 2L, 1L)) + 1L], collapse = "")
  }, character(1))
}

#' @title Bounds of geohash cells
#' @param hash A character vector of geohashes.
#' @return A data frame with columns `xmin`, `ymin`, `xmax` and `ymax` in
#'   degrees, one row per hash.
#' @export
geohash.bounds <- function(hash) {
  alphabet <- strsplit("0123456789bcdefghjkmnpqrstuvwxyz", "")[[1]]
  hash <- tolower(hash)
  out <- data.frame(xmin = -180, ymin = -90, xmax = 180, ymax = 90)[rep(1, length(hash)), ]
  for (i in seq_along(hash)) {
    idx <- match(strsplit(hash[i], "")[[1]], alphabet) - 1L
    if (anyNA(idx)) {
      stop(sprintf("\"%s\" is not a valid geohash.", hash[i]), call. = FALSE)
    }
    bits <- as.vector(vapply(idx, function(v) (v %/% c(16L, 8L, 4L, 2L, 1L)) %% 2L,
      integer(5)))
    lon.bits <- bits[c(TRUE, FALSE)]
    lat.bits <- bits[c(FALSE, TRUE)]

    # Narrow the longitude interval
    for (b in lon.bits) {
      mid <- (out$xmin[i] + out$xmax[i]) / 2
      if (b == 1L) out$xmin[i] <- mid else out$xmax[i] <- mid
    }

    # Narrow the latitude interval
    for (b in lat.bits) {
      mid <- (out$ymin[i] + out$ymax[i]) / 2
      if (b == 1L) out$ymin[i] <- mid else out$ymax[i] <- mid
    }
  }
  rownames(out) <- NULL
  out
}

#' @title Decode geohashes to cell centres
#' @param hash A character vector of geohashes.
#' @return A data frame with columns `lon`, `lat`, `lon.err` and `lat.err`,
#'   where the errors are half the cell width and height.
#' @export
geohash.decode <- function(hash) {
  b <- geohash.bounds(hash)

  # Along longitude
  lon <- (b$xmin + b$xmax) / 2
  lon.err <- (b$xmax - b$xmin) / 2

  # Along latitude
  lat <- (b$ymin + b$ymax) / 2
  lat.err <- (b$ymax - b$ymin) / 2
  data.frame(lon = lon, lat = lat, lon.err = lon.err, lat.err = lat.err)
}

#' @title Adjacent geohash in one direction
#' @param hash A character vector of geohashes.
#' @param direction One of `"north"`, `"south"`, `"east"`, `"west"`,
#'   `"northeast"`, `"northwest"`, `"southeast"` or `"southwest"`.
#' @return A character vector of geohashes with the same precision as
#'   `hash`; `NA` beyond a pole.
#' @export
geohash.neighbour <- function(hash, direction = c("north", "south", "east", "west",
                                                  "northeast", "northwest", "southeast",
                                                  "southwest")) {
  direction <- match.arg(direction)
  centre <- geohash.decode(hash)
  width <- 2 * centre$lon.err
  height <- 2 * centre$lat.err
  step <- switch(direction,
    north = c(0, 1),
    south = c(0, -1),
    east = c(1, 0),
    west = c(-1, 0),
    northeast = c(1, 1),
    northwest = c(-1, 1),
    southeast = c(1, -1),
    southwest = c(-1, -1)
  )
  lon <- centre$lon + step[1] * width
  lat <- centre$lat + step[2] * height
  lon <- ((lon + 180) %% 360) - 180
  out <- rep(NA_character_, length(hash))
  ok <- abs(lat) < 90
  out[ok] <- geohash.encode(lon[ok], lat[ok], precision = nchar(hash[1]))
  out
}

#' @title Cell size at a geohash precision
#' @param precision Number of characters.
#' @param lat Latitude in degrees at which to measure widths in metres.
#' @return A list with elements `bits`, `lon.bits`, `lat.bits`, `width.deg`,
#'   `height.deg`, `width.m` and `height.m`.
#' @keywords internal
geohash.cell.size <- function(precision, lat = 0) {
  metres.per.deg <- 2 * pi * 6371008.8 / 360
  bits <- 5L * precision
  lon.bits <- ceiling(bits / 2)
  lat.bits <- floor(bits / 2)
  width.deg <- 360 / 2^lon.bits
  height.deg <- 180 / 2^lat.bits
  width.m <- width.deg * metres.per.deg * cos(lat * pi / 180)
  height.m <- height.deg * metres.per.deg
  list(
    bits = bits,
    lon.bits = lon.bits,
    lat.bits = lat.bits,
    width.deg = width.deg,
    height.deg = height.deg,
    width.m = width.m,
    height.m = height.m
  )
}

#' @title Geohashes covering a bounding box
#' @param bbox A named numeric vector `c(xmin, ymin, xmax, ymax)`.
#' @param precision Number of characters per hash.
#' @param max.cells Refuse to return more than this many hashes.
#' @return A sorted character vector of unique geohashes.
#' @export
geohash.cover <- function(bbox, precision, max.cells = 5000) {
  size <- geohash.cell.size(precision)

  # Column centres along longitude
  x.from <- floor((bbox[["xmin"]] + 180) / size$width.deg)
  x.to <- floor((bbox[["xmax"]] + 180) / size$width.deg)
  xs <- -180 + (seq(x.from, x.to) + 0.5) * size$width.deg

  # Row centres along latitude
  y.from <- floor((bbox[["ymin"]] + 90) / size$height.deg)
  y.to <- floor((bbox[["ymax"]] + 90) / size$height.deg)
  ys <- -90 + (seq(y.from, y.to) + 0.5) * size$height.deg
  if (length(xs) * length(ys) > max.cells) {
    stop(sprintf("The box needs %d cells, more than `max.cells`.", length(xs) * length(ys)),
      call. = FALSE)
  }
  pts <- expand.grid(lon = xs, lat = ys)
  sort(unique(geohash.encode(pts$lon, pts$lat, precision)))
}
