#' Buffer parcels by a fixed distance
#'
#' Buffers each feature in a projected layer and records the buffer distance
#' in a new column.
#'
#' @param parcels An `sf` object of parcel polygons.
#' @param dist Buffer distance in map units.
#' @param segments Number of segments used per quarter circle.
#' @return An `sf` object of buffered polygons.
#' @family overlay helpers
#' @export
buffer_parcels <- function(parcels, dist, segments = 16) {
  assert_projected(parcels, "parcels")
  stopifnot(is.numeric(dist), length(dist) %in% c(1, nrow(parcels)))
  out <- sf::st_buffer(parcels, dist = dist, nQuadSegs = segments)
  out$buffer_m <- dist
  out
}

#' Intersect parcels with a zoning layer
#'
#' Splits parcels by zone boundaries and keeps the attributes of both layers.
#' Slivers smaller than `min_area` are dropped.
#'
#' @param parcels An `sf` object of parcel polygons.
#' @param zones An `sf` object of zone polygons.
#' @param min_area Smallest piece to keep, in square map units.
#' @return An `sf` object of parcel pieces with an `area_m2` column.
#' @family overlay helpers
#' @export
intersect_parcels <- function(parcels, zones, min_area = 1) {
  assert_projected(parcels, "parcels")
  assert_same_crs(parcels, zones)
  sf::st_agr(parcels) <- "constant"
  sf::st_agr(zones) <- "constant"
  pieces <- sf::st_intersection(parcels, zones)
  pieces <- pieces[sf::st_dimension(pieces) == 2, ]
  pieces$area_m2 <- as.numeric(sf::st_area(pieces))
  pieces[pieces$area_m2 >= min_area, ]
}

#' Join parcels to the zone containing their centroid
#'
#' @param parcels An `sf` object of parcel polygons.
#' @param zones An `sf` object of zone polygons.
#' @param zone_col Name of the zone identifier column in `zones`.
#' @return `parcels` with the zone identifier column added.
#' @family overlay helpers
#' @export
join_parcels_to_zones <- function(parcels, zones, zone_col = "zone_id") {
  assert_same_crs(parcels, zones)
  if (!zone_col %in% names(zones)) {
    stop("Column '", zone_col, "' not found in `zones`.", call. = FALSE)
  }
  centroids <- sf::st_point_on_surface(sf::st_geometry(parcels))
  hits <- sf::st_within(centroids, zones)
  idx <- vapply(hits, function(h) if (length(h) > 0) h[[1]] else NA_integer_, integer(1))
  parcels[[zone_col]] <- zones[[zone_col]][idx]
  parcels
}

#' Count points falling within each parcel
#'
#' @param parcels An `sf` object of parcel polygons.
#' @param points An `sf` object of point features.
#' @param name Name of the count column to add.
#' @return `parcels` with an integer count column added.
#' @family overlay helpers
#' @export
count_points_in_parcels <- function(parcels, points, name = "n_points") {
  assert_same_crs(parcels, points)
  hits <- sf::st_intersects(parcels, points)
  parcels[[name]] <- lengths(hits)
  parcels
}

#' Summarise parcel area by a grouping column
#'
#' Computes total, mean and maximum area per group, in hectares.
#'
#' @param parcels An `sf` object of parcel polygons.
#' @param by Name of the grouping column.
#' @return A data frame with one row per group.
#' @export
summarise_parcel_area <- function(parcels, by) {
  assert_projected(parcels, "parcels")
  area_ha <- as.numeric(sf::st_area(parcels)) / 10000
  groups <- sf::st_drop_geometry(parcels)[[by]]
  keys <- sort(unique(groups))
  out <- data.frame(
    group = keys,
    n = integer(length(keys)),
    total_ha = numeric(length(keys)),
    mean_ha = numeric(length(keys)),
    max_ha = numeric(length(keys))
  )
  for (i in seq_along(keys)) {
    a <- area_ha[groups == keys[i]]
    out$n[i] <- length(a)
    out$total_ha[i] <- sum(a)
    out$mean_ha[i] <- mean(a)
    out$max_ha[i] <- max(a)
  }
  names(out)[1] <- by
  out
}

#' Compute the share of each parcel covered by another layer
#'
#' For every parcel, returns the fraction of its area that overlaps the union
#' of the `cover` layer, for example a flood zone or protected area.
#'
#' @param parcels An `sf` object of parcel polygons.
#' @param cover An `sf` object of polygons to measure coverage against.
#' @return A numeric vector between 0 and 1, one value per parcel.
#' @export
coverage_share <- function(parcels, cover) {
  assert_projected(parcels, "parcels")
  assert_same_crs(parcels, cover)
  cover_union <- sf::st_union(sf::st_geometry(cover))
  parcel_geom <- sf::st_geometry(parcels)
  total <- as.numeric(sf::st_area(parcel_geom))
  share <- numeric(length(parcel_geom))
  touching <- lengths(sf::st_intersects(parcel_geom, cover_union)) > 0
  for (i in which(touching)) {
    piece <- sf::st_intersection(parcel_geom[i], cover_union)
    share[i] <- sum(as.numeric(sf::st_area(piece))) / total[i]
  }
  pmin(share, 1)
}

#' Dissolve parcels by an attribute
#'
#' @param parcels An `sf` object of parcel polygons.
#' @param by Name of the grouping column.
#' @return An `sf` object with one multipolygon per group.
#' @export
dissolve_parcels <- function(parcels, by) {
  geom <- sf::st_geometry(parcels)
  keys <- parcels[[by]]
  pieces <- lapply(split(seq_along(geom), keys), function(idx) {
    sf::st_union(geom[idx])
  })
  out <- data.frame(names(pieces), stringsAsFactors = FALSE)
  names(out) <- by
  sf::st_sf(out, geometry = do.call(c, pieces), crs = sf::st_crs(parcels))
}
