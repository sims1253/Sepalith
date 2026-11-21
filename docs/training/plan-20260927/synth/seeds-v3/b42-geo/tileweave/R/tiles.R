#' Slippy-map tile containing each point
#'
#' @param lon Numeric vector of longitudes in degrees.
#' @param lat Numeric vector of latitudes in degrees.
#' @param zoom A single integer zoom level between 0 and 24.
#' @return A data frame with integer columns `x`, `y` and `zoom`.
#' @export
lonLatToTile = function(lon, lat, zoom) {
  # Check `lon`
  if (!is.numeric(lon) || any(abs(lon) > 180, na.rm = TRUE)) {
    stop(sprintf("`%s` is not a valid %s.", "lon", "longitude"), call. = FALSE)
  }

  # Check `lat`
  if (!is.numeric(lat) || any(abs(lat) > 90, na.rm = TRUE)) {
    stop(sprintf("`%s` is not a valid %s.", "lat", "latitude"), call. = FALSE)
  }

  # Check `zoom`
  if (length(zoom) != 1 || is.na(zoom) || zoom < 0 || zoom > 24 || zoom %% 1 != 0) {
    stop(sprintf("`%s` is not a valid %s.", "zoom", "zoom level"), call. = FALSE)
  }
  n = 2^zoom
  maxLat = atan(sinh(pi)) * 180 / pi
  phi = pmin(pmax(lat, -maxLat), maxLat) * pi / 180
  x = floor((lon + 180) / 360 * n)
  y = floor((1 - log(tan(phi) + 1 / cos(phi)) / pi) / 2 * n)
  x = pmin(pmax(x, 0), n - 1)
  y = pmin(pmax(y, 0), n - 1)
  return(data.frame(x = as.integer(x), y = as.integer(y), zoom = as.integer(zoom)))
}

#' Coordinates of a tile corner or centre
#'
#' @param x,y Integer tile indices.
#' @param zoom Zoom level.
#' @param corner One of `"northwest"`, `"northeast"`, `"southwest"`,
#'   `"southeast"` or `"center"`.
#' @return A data frame with columns `lon` and `lat` in degrees.
#' @export
tileToLonLat = function(x, y, zoom,
                        corner = c("northwest", "northeast", "southwest", "southeast",
                                   "center")) {
  corner = match.arg(corner)
  n = 2^zoom
  offset = switch(corner,
    northwest = {
      dx = 0
      dy = 0
      c(dx, dy)
    },
    northeast = {
      dx = 1
      dy = 0
      c(dx, dy)
    },
    southwest = {
      dx = 0
      dy = 1
      c(dx, dy)
    },
    southeast = {
      dx = 1
      dy = 1
      c(dx, dy)
    },
    center = {
      dx = 0.5
      dy = 0.5
      c(dx, dy)
    }
  )
  px = (x + offset[1]) / n
  py = (y + offset[2]) / n
  lon = px * 360 - 180
  lat = atan(sinh(pi * (1 - 2 * py))) * 180 / pi
  return(data.frame(lon = lon, lat = lat))
}

#' Geographic bounds of a tile
#'
#' @param x,y Integer tile indices.
#' @param zoom Zoom level.
#' @param crs `"lonlat"` for degrees or `"webmercator"` for metres.
#' @return A data frame with columns `xmin`, `ymin`, `xmax` and `ymax`, one
#'   row per tile.
#' @export
tileBounds = function(x, y, zoom, crs = c("lonlat", "webmercator")) {
  crs = match.arg(crs)
  n = 2^zoom
  if (any(x < 0 | x >= n | y < 0 | y >= n)) {
    stop(sprintf("Tile indices must lie in [0, %d] at zoom %d.", n - 1, zoom),
      call. = FALSE)
  }
  if (crs == "lonlat") {
    # West and east edges
    west = x / n * 360 - 180
    east = (x + 1) / n * 360 - 180

    # North and south edges
    north = atan(sinh(pi * (1 - 2 * y / n))) * 180 / pi
    south = atan(sinh(pi * (1 - 2 * (y + 1) / n))) * 180 / pi
  } else {
    halfWorld = pi * 6378137

    # West and east edges
    west = x / n * 2 * halfWorld - halfWorld
    east = (x + 1) / n * 2 * halfWorld - halfWorld

    # North and south edges
    north = halfWorld - y / n * 2 * halfWorld
    south = halfWorld - (y + 1) / n * 2 * halfWorld
  }
  return(data.frame(xmin = west, ymin = south, xmax = east, ymax = north))
}

#' Tiles related to a given tile
#'
#' @param x,y Integer tile indices of a single tile.
#' @param zoom Zoom level.
#' @param relation One of `"parent"`, `"children"`, `"siblings"` or
#'   `"neighbours"`.
#' @return A data frame with integer columns `x`, `y` and `zoom`.
#' @export
tileRelatives = function(x, y, zoom,
                         relation = c("parent", "children", "siblings", "neighbours")) {
  relation = match.arg(relation)
  stopifnot(length(x) == 1, length(y) == 1)
  n = 2^zoom
  out = switch(relation,
    parent = {
      if (zoom == 0) {
        stop(sprintf("Zoom level 0 has no %s.", relation), call. = FALSE)
      }
      data.frame(x = x %/% 2, y = y %/% 2, zoom = zoom - 1)
    },
    children = {
      dx = c(0, 1, 0, 1)
      dy = c(0, 0, 1, 1)
      data.frame(x = 2 * x + dx, y = 2 * y + dy, zoom = zoom + 1)
    },
    siblings = {
      if (zoom == 0) {
        stop(sprintf("Zoom level 0 has no %s.", relation), call. = FALSE)
      }
      dx = c(0, 1, 0, 1)
      dy = c(0, 0, 1, 1)
      keep = !(2 * (x %/% 2) + dx == x & 2 * (y %/% 2) + dy == y)
      data.frame(x = 2 * (x %/% 2) + dx[keep], y = 2 * (y %/% 2) + dy[keep], zoom = zoom)
    },
    neighbours = {
      dx = c(-1, 0, 1, -1, 1, -1, 0, 1)
      dy = c(-1, -1, -1, 0, 0, 1, 1, 1)
      # Wrap around the antimeridian, stop at the poles
      nx = (x + dx) %% n
      ny = y + dy
      keep = ny >= 0 & ny < n & !(nx == x & ny == y)
      data.frame(x = nx[keep], y = ny[keep], zoom = zoom)
    }
  )
  out[] = lapply(out, as.integer)
  return(out)
}

#' All tiles covering a bounding box
#'
#' @param bbox A named numeric vector `c(xmin, ymin, xmax, ymax)` in degrees.
#' @param zoom Zoom level.
#' @param maxTiles Refuse to return more than this many tiles.
#' @return A data frame with integer columns `x`, `y` and `zoom`.
#' @export
tilesInBbox = function(bbox, zoom, maxTiles = 10000) {
  sides = c("xmin", "ymin", "xmax", "ymax")
  if (!all(sides %in% names(bbox))) {
    stop("`bbox` must have elements xmin, ymin, xmax and ymax.", call. = FALSE)
  }

  # Upper-left tile: west edge, north edge
  upperLeft = lonLatToTile(bbox[["xmin"]], bbox[["ymax"]], zoom)
  xFrom = upperLeft$x
  yFrom = upperLeft$y

  # Lower-right tile: east edge, south edge
  lowerRight = lonLatToTile(bbox[["xmax"]], bbox[["ymin"]], zoom)
  xTo = lowerRight$x
  yTo = lowerRight$y
  if (xFrom > xTo) {
    xs = c(seq(xFrom, 2^zoom - 1), seq(0, xTo))
  } else {
    xs = seq(xFrom, xTo)
  }
  ys = seq(yFrom, yTo)
  if (length(xs) * length(ys) > maxTiles) {
    stop(sprintf("The box needs %d tiles, more than `maxTiles`.", length(xs) * length(ys)),
      call. = FALSE)
  }
  grid = expand.grid(x = xs, y = ys)
  return(data.frame(x = as.integer(grid$x), y = as.integer(grid$y),
    zoom = as.integer(zoom)))
}

#' Ground resolution and scale of a zoom level
#'
#' @param lat Latitude in degrees at which to measure.
#' @param zoom Zoom level.
#' @param tileSize Tile edge in pixels.
#' @param dpi Screen resolution used for the scale denominator.
#' @return A list with elements `tilesPerSide`, `pixelsPerSide`,
#'   `metersPerPixel`, `tileWidth` (metres), `scaleDenominator`, `tileCount`
#'   and `tileArea` (square kilometres).
#' @export
tileMetrics = function(lat, zoom, tileSize = 256, dpi = 96) {
  earthCircumference = 2 * pi * 6378137
  inchesPerMeter = 1 / 0.0254

  # 1. tilesPerSide
  tilesPerSide = 2^zoom

  # 2. pixelsPerSide
  pixelsPerSide = tileSize * tilesPerSide

  # 3. metersPerPixel
  metersPerPixel = cos(lat * pi / 180) * earthCircumference / pixelsPerSide

  # 4. tileWidth
  tileWidth = metersPerPixel * tileSize

  # 5. scaleDenominator
  scaleDenominator = metersPerPixel * dpi * inchesPerMeter

  # 6. tileCount
  tileCount = tilesPerSide^2

  # 7. tileArea
  tileArea = tileWidth^2 / 1e6
  return(list(
    tilesPerSide = tilesPerSide,
    pixelsPerSide = pixelsPerSide,
    metersPerPixel = metersPerPixel,
    tileWidth = tileWidth,
    scaleDenominator = scaleDenominator,
    tileCount = tileCount,
    tileArea = tileArea
  ))
}
