#' Breadth-first search from a source vertex
#'
#' Visits vertices in order of increasing hop distance and records the
#' discovery order, hop distance and BFS parent of each reachable vertex.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()].
#' @param source Name of the starting vertex.
#' @param max_depth Maximum number of hops to explore; `Inf` for no limit.
#' @return A data frame with columns `vertex`, `order`, `depth` and `parent`.
#' @export
#' @family traversal
bfs <- function(adj, source, max_depth = Inf) {
  if (!source %in% names(adj)) {
    stop("Unknown source vertex: ", source, call. = FALSE)
  }
  n <- length(adj)
  depth <- setNames(rep(NA_integer_, n), names(adj))
  parent <- setNames(rep(NA_character_, n), names(adj))
  visited <- character(0)
  queue <- source
  depth[source] <- 0L
  while (length(queue) > 0) {
    v <- queue[1]
    queue <- queue[-1]
    visited <- c(visited, v)
    if (depth[v] >= max_depth) {
      next
    }
    for (u in adj[[v]]) {
      if (is.na(depth[u])) {
        depth[u] <- depth[v] + 1L
        parent[u] <- v
        queue <- c(queue, u)
      }
    }
  }
  out <- data.frame(
    vertex = visited,
    order = seq_along(visited),
    depth = unname(depth[visited]),
    parent = unname(parent[visited]),
    stringsAsFactors = FALSE
  )
  return(out)
}

#' Depth-first search from a source vertex
#'
#' Iterative DFS using an explicit stack. Neighbours are pushed in reverse
#' so that they are visited in adjacency-list order.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()].
#' @param source Name of the starting vertex.
#' @param max_depth Maximum number of hops to explore; `Inf` for no limit.
#' @return A data frame with columns `vertex`, `order`, `depth` and `parent`.
#' @export
#' @family traversal
dfs <- function(adj, source, max_depth = Inf) {
  if (!source %in% names(adj)) {
    stop("Unknown source vertex: ", source, call. = FALSE)
  }
  n <- length(adj)
  depth <- setNames(rep(NA_integer_, n), names(adj))
  parent <- setNames(rep(NA_character_, n), names(adj))
  visited <- character(0)
  stack <- list(list(v = source, d = 0L, p = NA_character_))
  while (length(stack) > 0) {
    top <- stack[[length(stack)]]
    stack[[length(stack)]] <- NULL
    if (top$v %in% visited) {
      next
    }
    visited <- c(visited, top$v)
    depth[top$v] <- top$d
    parent[top$v] <- top$p
    if (top$d >= max_depth) {
      next
    }
    for (u in rev(adj[[top$v]])) {
      if (!u %in% visited) {
        stack[[length(stack) + 1]] <- list(v = u, d = top$d + 1L, p = top$v)
      }
    }
  }
  out <- data.frame(
    vertex = visited,
    order = seq_along(visited),
    depth = unname(depth[visited]),
    parent = unname(parent[visited]),
    stringsAsFactors = FALSE
  )
  return(out)
}

#' Reconstruct a path from a parent table
#'
#' Walks parent pointers back from `target` to the traversal root.
#'
#' @param walk A data frame as returned by [bfs()] or [dfs()].
#' @param target Name of the vertex to walk back from.
#' @return A character vector of vertex names from the root to `target`, or
#'   `character(0)` if `target` was not reached.
#' @export
#' @family traversal
walk_path <- function(walk, target) {
  if (!target %in% walk$vertex) {
    return(character(0))
  }
  parents <- setNames(walk$parent, walk$vertex)
  path <- target
  current <- target
  while (!is.na(parents[[current]])) {
    current <- parents[[current]]
    path <- c(current, path)
  }
  return(path)
}

#' Label connected components of an undirected graph
#'
#' Runs repeated BFS sweeps, assigning a component id to every vertex that
#' has not yet been labelled.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()].
#' @return A named integer vector of component ids, numbered from 1 in order
#'   of first discovery.
#' @export
#' @family components
connected_components <- function(adj) {
  comp <- setNames(rep(NA_integer_, length(adj)), names(adj))
  next_id <- 0L
  for (start in names(adj)) {
    if (!is.na(comp[start])) {
      next
    }
    next_id <- next_id + 1L
    queue <- start
    comp[start] <- next_id
    while (length(queue) > 0) {
      v <- queue[1]
      queue <- queue[-1]
      for (u in adj[[v]]) {
        if (is.na(comp[u])) {
          comp[u] <- next_id
          queue <- c(queue, u)
        }
      }
    }
  }
  return(comp)
}

#' Summarise component sizes
#'
#' @param comp A named integer vector as returned by [connected_components()].
#' @return A data frame with columns `component`, `size` and `share`, sorted by
#'   decreasing size.
#' @export
#' @family components
component_sizes <- function(comp) {
  tab <- table(comp)
  out <- data.frame(
    component = as.integer(names(tab)),
    size = as.integer(tab),
    stringsAsFactors = FALSE
  )
  out$share <- out$size / sum(out$size)
  out <- out[order(-out$size, out$component), , drop = FALSE]
  rownames(out) <- NULL
  return(out)
}

#' Test whether an undirected graph is bipartite
#'
#' Attempts a two-colouring with BFS. Returns the colouring when one exists.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()].
#' @return A list with `bipartite` (logical) and `side` (named integer vector of
#'   0/1 colours, or `NULL` when the graph is not bipartite).
#' @export
#' @family components
is_bipartite <- function(adj) {
  side <- setNames(rep(NA_integer_, length(adj)), names(adj))
  for (start in names(adj)) {
    if (!is.na(side[start])) {
      next
    }
    side[start] <- 0L
    queue <- start
    while (length(queue) > 0) {
      v <- queue[1]
      queue <- queue[-1]
      for (u in adj[[v]]) {
        if (is.na(side[u])) {
          side[u] <- 1L - side[v]
          queue <- c(queue, u)
        } else if (side[u] == side[v]) {
          return(list(bipartite = FALSE, side = NULL))
        }
      }
    }
  }
  return(list(bipartite = TRUE, side = side))
}
