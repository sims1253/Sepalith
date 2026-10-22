#' Single-source shortest paths with Dijkstra's algorithm
#'
#' Uses a simple linear scan instead of a heap, which is fine for the small
#' graphs this package targets (a few thousand vertices).
#'
#' @param wadj A weighted adjacency list as returned by
#'   [edges_to_weighted_adjlist()].
#' @param source Name of the starting vertex.
#' @return A list with named numeric `dist` and named character `prev`.
#' @export
#' @family shortest-paths
dijkstra <- function(wadj, source) {
  if (!source %in% names(wadj)) {
    stop("Unknown source vertex: ", source, call. = FALSE)
  }
  v <- names(wadj)
  dist <- setNames(rep(Inf, length(v)), v)
  prev <- setNames(rep(NA_character_, length(v)), v)
  done <- setNames(rep(FALSE, length(v)), v)
  dist[source] <- 0
  repeat {
    open <- v[!done]
    if (length(open) == 0) {
      break
    }
    u <- open[which.min(dist[open])]
    if (is.infinite(dist[u])) {
      break
    }
    done[u] <- TRUE
    nb <- wadj[[u]]
    for (w in names(nb)) {
      alt <- dist[u] + nb[[w]]
      if (alt < dist[w]) {
        dist[w] <- alt
        prev[w] <- u
      }
    }
  }
  return(list(dist = dist, prev = prev))
}

#' Single-source shortest paths with Bellman-Ford
#'
#' Handles negative edge weights and reports negative cycles reachable from
#' the source.
#'
#' @param wadj A weighted adjacency list as returned by
#'   [edges_to_weighted_adjlist()].
#' @param source Name of the starting vertex.
#' @return A list with named numeric `dist` and named character `prev`.
#' @export
#' @family shortest-paths
bellman_ford <- function(wadj, source) {
  if (!source %in% names(wadj)) {
    stop("Unknown source vertex: ", source, call. = FALSE)
  }
  v <- names(wadj)
  dist <- setNames(rep(Inf, length(v)), v)
  prev <- setNames(rep(NA_character_, length(v)), v)
  dist[source] <- 0
  for (iter in seq_len(length(v) - 1L)) {
    changed <- FALSE
    for (u in v) {
      if (is.infinite(dist[u])) {
        next
      }
      nb <- wadj[[u]]
      for (w in names(nb)) {
        alt <- dist[u] + nb[[w]]
        if (alt < dist[w]) {
          dist[w] <- alt
          prev[w] <- u
          changed <- TRUE
        }
      }
    }
    if (!changed) {
      break
    }
  }
  for (u in v[is.finite(dist)]) {
    nb <- wadj[[u]]
    for (w in names(nb)) {
      if (dist[u] + nb[[w]] < dist[w]) {
        stop("Graph contains a negative cycle reachable from ", source, call. = FALSE)
      }
    }
  }
  return(list(dist = dist, prev = prev))
}

#' Extract a shortest path from a predecessor vector
#'
#' @param sp A list as returned by [dijkstra()] or [bellman_ford()].
#' @param target Name of the destination vertex.
#' @return A character vector of vertices from source to `target`, or
#'   `character(0)` if `target` is unreachable.
#' @export
#' @family shortest-paths
shortest_path_to <- function(sp, target) {
  if (is.infinite(sp$dist[[target]])) {
    return(character(0))
  }
  path <- target
  while (!is.na(sp$prev[[path[1]]])) {
    path <- c(sp$prev[[path[1]]], path)
  }
  return(path)
}

#' All-pairs shortest distances with Floyd-Warshall
#'
#' @param m A square numeric matrix of edge weights, with `Inf` (or `NA`) for
#'   missing edges.
#' @return A square numeric matrix of shortest distances with the same
#'   dimnames as `m`.
#' @export
#' @family shortest-paths
floyd_warshall <- function(m) {
  if (nrow(m) != ncol(m)) {
    stop("Weight matrix must be square.", call. = FALSE)
  }
  d <- m
  d[is.na(d)] <- Inf
  diag(d) <- pmin(diag(d), 0)
  n <- nrow(d)
  for (k in seq_len(n)) {
    for (i in seq_len(n)) {
      if (is.infinite(d[i, k])) {
        next
      }
      via <- d[i, k] + d[k, ]
      better <- via < d[i, ]
      d[i, better] <- via[better]
    }
  }
  if (any(diag(d) < 0)) {
    warning("Negative cycle detected; distances are not well defined.", call. = FALSE)
  }
  return(d)
}

#' Eccentricity of every vertex
#'
#' The greatest shortest-path distance from a vertex to any other reachable
#' vertex.
#'
#' @param d A distance matrix as returned by [floyd_warshall()].
#' @return A named numeric vector of eccentricities.
#' @export
#' @family distance-summaries
eccentricity <- function(d) {
  d[is.infinite(d)] <- NA
  ecc <- apply(d, 1, max, na.rm = TRUE)
  return(ecc)
}

#' Closeness of every vertex
#'
#' Reciprocal of the mean shortest-path distance to all reachable vertices.
#'
#' @param d A distance matrix as returned by [floyd_warshall()].
#' @return A named numeric vector of closeness scores.
#' @export
#' @family distance-summaries
closeness_from_dist <- function(d) {
  d[is.infinite(d)] <- NA
  diag(d) <- NA
  avg <- apply(d, 1, mean, na.rm = TRUE)
  clo <- ifelse(is.nan(avg), 0, 1 / avg)
  return(clo)
}

#' Diameter and radius of a graph
#'
#' @param d A distance matrix as returned by [floyd_warshall()].
#' @return A list with numeric `diameter`, numeric `radius` and character
#'   `center` (the vertices achieving the radius).
#' @export
#' @family distance-summaries
graph_extent <- function(d) {
  ecc <- eccentricity(d)
  diameter <- max(ecc)
  radius <- min(ecc)
  center <- names(ecc)[ecc == radius]
  return(list(diameter = diameter, radius = radius, center = center))
}
