#' @title Find the nearest facility for each origin
#'
#' @description
#' Uses straight-line distance to match every origin to its closest facility.
#'
#' @param origins An `sf` object of origin points.
#' @param facilities An `sf` object of facility points.
#' @param id_col Name of the facility identifier column.
#' @return `origins` with `facility_id` and `dist_m` columns added.
#' @family nearest-facility helpers
#' @export
nearest_facility <- function(origins, facilities, id_col = "facility_id") {
  assert_same_crs(origins, facilities)
  assert_projected(origins, "origins")

  idx <- sf::st_nearest_feature(origins, facilities)
  dist <- sf::st_distance(origins, facilities[idx, ], by_element = TRUE)

  origins$facility_id <- facilities[[id_col]][idx]
  origins$dist_m <- as.numeric(dist)
  return(origins)
}

#' @title Find the k nearest facilities for each origin
#'
#' @description
#' Returns a long table with one row per origin and candidate facility, ranked
#' by straight-line distance.
#'
#' @param origins An `sf` object of origin points.
#' @param facilities An `sf` object of facility points.
#' @param k Number of facilities to keep per origin.
#' @param id_col Name of the facility identifier column.
#' @return A data frame with `origin`, `rank`, `facility_id` and `dist_m`.
#' @family nearest-facility helpers
#' @export
nearest_k_facilities <- function(origins, facilities, k = 3, id_col = "facility_id") {
  assert_same_crs(origins, facilities)
  assert_projected(origins, "origins")
  k <- min(k, nrow(facilities))

  # The full matrix is fine for the few hundred facilities we deal with
  dmat <- units::drop_units(sf::st_distance(origins, facilities))
  n <- nrow(origins)
  out <- data.frame(
    origin = rep(seq_len(n), each = k),
    rank = rep(seq_len(k), times = n),
    facility_id = NA,
    dist_m = NA_real_
  )

  for (i in seq_len(n)) {
    ord <- order(dmat[i, ])[seq_len(k)]
    rows <- (i - 1) * k + seq_len(k)
    out$facility_id[rows] <- facilities[[id_col]][ord]
    out$dist_m[rows] <- dmat[i, ord]
  }

  return(out)
}

#' @title Find facilities within a service radius
#'
#' @param origins An `sf` object of origin points.
#' @param facilities An `sf` object of facility points.
#' @param radius Service radius in map units.
#' @return `origins` with an integer `n_within` column added.
#' @family nearest-facility helpers
#' @export
facilities_within <- function(origins, facilities, radius) {
  assert_same_crs(origins, facilities)
  assert_projected(origins, "origins")
  hits <- sf::st_is_within_distance(origins, facilities, dist = radius)
  origins$n_within <- lengths(hits)
  return(origins)
}

#' @title Snap points to the nearest network edge
#'
#' @description
#' Moves each point onto the closest line of a road network, which is useful
#' before computing network distances.
#'
#' @param points An `sf` object of point features.
#' @param network An `sf` object of road centrelines.
#' @param max_dist Points further than this from any edge are left unchanged.
#' @return `points` with snapped geometries and a `snap_dist_m` column.
#' @export
snap_to_network <- function(points, network, max_dist = Inf) {
  assert_same_crs(points, network)
  edge_idx <- sf::st_nearest_feature(points, network)
  geom <- sf::st_geometry(points)
  snap_dist <- numeric(length(geom))

  for (i in seq_along(geom)) {
    # st_nearest_points() gives a two-point line from the point to the edge;
    # its second vertex is the snapped location
    link <- sf::st_nearest_points(geom[i], sf::st_geometry(network)[edge_idx[i]])
    snap_dist[i] <- as.numeric(sf::st_length(link))
    if (snap_dist[i] <= max_dist) {
      geom[i] <- sf::st_cast(link, "POINT")[2]
    }
  }

  sf::st_geometry(points) <- geom
  points$snap_dist_m <- snap_dist
  return(points)
}

#' @title Build an origin-destination travel-time matrix
#'
#' @description
#' Converts straight-line distances to travel times using a constant speed and
#' a detour factor that approximates the road network.
#'
#' @param origins An `sf` object of origin points.
#' @param facilities An `sf` object of facility points.
#' @param speed_kmh Average travel speed in kilometres per hour.
#' @param detour Ratio of network distance to straight-line distance.
#' @return A numeric matrix of travel times in minutes.
#' @export
travel_time_matrix <- function(origins, facilities, speed_kmh = 40, detour = 1.3) {
  if (speed_kmh <= 0) {
    stop("`speed_kmh` must be positive.", call. = FALSE)
  }
  if (detour < 1) {
    stop("`detour` must be at least 1.", call. = FALSE)
  }
  assert_same_crs(origins, facilities)
  assert_projected(origins, "origins")

  dmat <- units::drop_units(sf::st_distance(origins, facilities))
  # metres -> km -> hours -> minutes
  minutes <- dmat * detour / 1000 / speed_kmh * 60
  dimnames(minutes) <- list(rownames(origins), rownames(facilities))
  return(minutes)
}

#' @title Summarise access to facilities by travel-time band
#'
#' @description
#' Origins whose travel times are all missing are counted in no band, but
#' still count towards the denominator of `share`.
#'
#' @param minutes A travel-time matrix as returned by [travel_time_matrix()].
#' @param bands Upper limits of the time bands, in minutes.
#' @return A data frame with the number and share of origins per band.
#' @export
access_by_band <- function(minutes, bands = c(10, 20, 30, 60)) {
  if (ncol(minutes) == 0) {
    stop("`minutes` has no facility columns.", call. = FALSE)
  }
  bands <- sort(unique(bands))

  # Only the closest facility matters for access; suppressWarnings() covers
  # rows that are entirely NA, which min() turns into Inf
  best <- suppressWarnings(apply(minutes, 1, min, na.rm = TRUE))
  best[is.infinite(best)] <- NA

  labels <- c(paste0("<=", bands), paste0(">", max(bands)))
  band <- cut(best, breaks = c(-Inf, bands, Inf), labels = labels, right = TRUE)
  counts <- as.integer(table(band))

  result <- data.frame(band = labels, n = counts, share = counts / length(best))
  return(result)
}
