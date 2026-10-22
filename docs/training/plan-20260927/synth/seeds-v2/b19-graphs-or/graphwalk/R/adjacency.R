#' Build an adjacency list from an edge table
#'
#' Converts a two-column edge table into a named list where each element
#' holds the neighbours of one vertex. Isolated vertices can be supplied.
#'
#' @param edges A data frame of edges with columns `from` and `to`.
#' @param vertices Optional character vector of vertex names to include even
#'   when they have no incident edges.
#' @param directed Logical; if `FALSE`, every edge is stored in both directions.
#' @return A named list of character vectors, one per vertex.
#' @export
edges_to_adjlist <- function(edges, vertices = NULL, directed = FALSE) {
  if (!all(c("from", "to") %in% names(edges))) {
    stop("`edges` must have columns `from` and `to`.", call. = FALSE)
  }

  from <- as.character(edges$from)
  to <- as.character(edges$to)
  all_v <- unique(c(vertices, from, to))

  # Start every vertex with an empty neighbour set so isolated ones survive
  adj <- setNames(vector("list", length(all_v)), all_v)
  for (v in all_v) {
    adj[[v]] <- character(0)
  }

  for (i in seq_along(from)) {
    adj[[from[i]]] <- c(adj[[from[i]]], to[i])
    # Self-loops are only stored once, even for undirected graphs
    if (!directed && from[i] != to[i]) {
      adj[[to[i]]] <- c(adj[[to[i]]], from[i])
    }
  }
  return(adj)
}

#' Build a weighted adjacency list from an edge table
#'
#' Like [edges_to_adjlist()] but keeps edge weights, storing each neighbour
#' set as a named numeric vector keyed by neighbour name.
#'
#' @param edges A data frame of edges with columns `from` and `to`.
#' @param weight Name of the column in `edges` holding numeric weights.
#' @param directed Logical; if `FALSE`, every edge is stored in both directions.
#' @return A named list of named numeric vectors, one per vertex.
#' @export
edges_to_weighted_adjlist <- function(edges, weight = "weight", directed = FALSE) {
  if (!all(c("from", "to", weight) %in% names(edges))) {
    stop("`edges` must have columns `from`, `to` and `", weight, "`.", call. = FALSE)
  }

  from <- as.character(edges$from)
  to <- as.character(edges$to)
  w <- as.numeric(edges[[weight]])
  if (anyNA(w) || any(w < 0)) {
    stop("Edge weights must be non-negative and non-missing.", call. = FALSE)
  }

  all_v <- unique(c(from, to))
  adj <- setNames(vector("list", length(all_v)), all_v)
  for (v in all_v) {
    adj[[v]] <- setNames(numeric(0), character(0))
  }

  # A repeated edge overwrites the earlier weight rather than adding to it
  for (i in seq_along(from)) {
    adj[[from[i]]][to[i]] <- w[i]
    if (!directed && from[i] != to[i]) {
      adj[[to[i]]][from[i]] <- w[i]
    }
  }
  return(adj)
}

#' Convert an adjacency list to an adjacency matrix
#'
#' Produces a square 0/1 matrix with vertices in the order of the list names.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()].
#' @return A square integer matrix with dimnames set to the vertex names.
#' @export
adjlist_to_matrix <- function(adj) {
  v <- names(adj)
  m <- matrix(0L, nrow = length(v), ncol = length(v), dimnames = list(v, v))
  for (from in v) {
    nb <- adj[[from]]
    if (length(nb) > 0) {
      m[from, nb] <- 1L
    }
  }
  return(m)
}

#' Convert an adjacency matrix to an adjacency list
#'
#' Reads non-zero entries row by row. Row names are used as vertex names,
#' falling back to integer labels when absent. Missing entries are treated
#' as absent edges.
#'
#' @param m A square numeric matrix; non-zero entries denote edges.
#' @return A named list of character vectors, one per vertex.
#' @export
matrix_to_adjlist <- function(m) {
  if (nrow(m) != ncol(m)) {
    stop("Adjacency matrix must be square.", call. = FALSE)
  }

  v <- rownames(m)
  if (is.null(v)) {
    v <- as.character(seq_len(nrow(m)))
  }

  adj <- setNames(vector("list", length(v)), v)
  for (i in seq_along(v)) {
    # which() drops NA cells instead of producing NA neighbour names
    adj[[i]] <- v[which(m[i, ] != 0)]
  }
  return(adj)
}

#' Vertex degrees
#'
#' `out_degree()` counts outgoing edges and `in_degree()` counts incoming
#' edges of every vertex in an adjacency list.
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()].
#' @return A named integer vector of degrees.
#' @rdname degree
#' @export
out_degree <- function(adj) {
  deg <- vapply(adj, length, integer(1))
  return(deg)
}

#' @rdname degree
#' @export
in_degree <- function(adj) {
  deg <- setNames(integer(length(adj)), names(adj))
  for (nb in adj) {
    for (u in nb) {
      deg[u] <- deg[u] + 1L
    }
  }
  return(deg)
}

#' Reverse every edge of a directed adjacency list
#'
#' @param adj A named adjacency list as returned by [edges_to_adjlist()].
#' @return A named adjacency list with the same vertices and reversed edges.
#' @export
transpose_adjlist <- function(adj) {
  rev_adj <- setNames(vector("list", length(adj)), names(adj))
  for (v in names(adj)) {
    rev_adj[[v]] <- character(0)
  }

  for (from in names(adj)) {
    for (to in adj[[from]]) {
      rev_adj[[to]] <- c(rev_adj[[to]], from)
    }
  }
  return(rev_adj)
}
