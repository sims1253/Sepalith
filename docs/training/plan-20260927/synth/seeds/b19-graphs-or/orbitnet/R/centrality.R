#' Compute one centrality measure
#'
#' Thin dispatcher over igraph's centrality functions so callers can pick a
#' measure by name and always get a named numeric vector back.
#'
#' @param g An `igraph` object.
#' @param measure Which centrality to compute.
#' @param weighted Logical; use the `weight` edge attribute when present.
#' @return A named numeric vector with one value per vertex.
#' @export
#' @family centrality
centrality_one <- function(g,
                           measure = c("degree", "strength", "betweenness",
                                       "closeness", "eigen", "pagerank"),
                           weighted = TRUE) {
  measure <- match.arg(measure)
  w <- if (weighted && "weight" %in% igraph::edge_attr_names(g)) {
    igraph::E(g)$weight
  } else {
    NULL
  }
  vals <- switch(measure,
    degree = igraph::degree(g),
    strength = igraph::strength(g, weights = w),
    betweenness = igraph::betweenness(g, weights = w, normalized = TRUE),
    closeness = igraph::closeness(g, weights = w, normalized = TRUE),
    eigen = igraph::eigen_centrality(g, weights = w)$vector,
    pagerank = igraph::page_rank(g, weights = w)$vector
  )
  stats::setNames(as.numeric(vals), igraph::V(g)$name)
}

#' Centrality table for several measures
#'
#' @param g An `igraph` object.
#' @param measures Character vector of measure names accepted by
#'   [centrality_one()].
#' @param weighted Logical; use the `weight` edge attribute when present.
#' @return A tibble with a `name` column and one column per measure.
#' @export
#' @family centrality
centrality_table <- function(g,
                             measures = c("degree", "betweenness", "pagerank"),
                             weighted = TRUE) {
  out <- tibble::tibble(name = igraph::V(g)$name)
  for (m in measures) {
    out[[m]] <- centrality_one(g, m, weighted = weighted)
  }
  out
}

#' Rank vertices within each centrality measure
#'
#' Converts a wide centrality table to long form and adds the within-measure
#' rank and percentile.
#'
#' @param tbl A tibble as returned by [centrality_table()].
#' @param ties How to break ties, passed to [base::rank()].
#' @return A long tibble with columns `name`, `measure`, `value`, `rank` and
#'   `percentile`.
#' @export
#' @family centrality
centrality_ranks <- function(tbl, ties = c("min", "average", "first")) {
  ties <- match.arg(ties)
  tbl %>%
    tidyr::pivot_longer(-name, names_to = "measure", values_to = "value") %>%
    dplyr::group_by(measure) %>%
    dplyr::mutate(
      rank = rank(-value, ties.method = ties),
      percentile = 1 - (rank - 1) / dplyr::n()
    ) %>%
    dplyr::ungroup() %>%
    dplyr::arrange(measure, rank)
}

#' Top vertices per centrality measure
#'
#' @param tbl A tibble as returned by [centrality_table()].
#' @param n Number of vertices to keep per measure.
#' @return A long tibble with the `n` highest-ranked vertices per measure.
#' @export
#' @family centrality
top_central <- function(tbl, n = 10) {
  centrality_ranks(tbl) %>%
    dplyr::group_by(measure) %>%
    dplyr::slice_min(rank, n = n, with_ties = FALSE) %>%
    dplyr::ungroup()
}

#' Spearman agreement between centrality measures
#'
#' @param tbl A tibble as returned by [centrality_table()].
#' @return A symmetric correlation matrix with one row per measure.
#' @export
#' @family centrality
centrality_agreement <- function(tbl) {
  vals <- tbl %>%
    dplyr::select(-name) %>%
    as.matrix()
  if (ncol(vals) < 2) {
    stop("Need at least two measures to compare.", call. = FALSE)
  }
  stats::cor(vals, method = "spearman", use = "pairwise.complete.obs")
}

#' Flag brokers: high betweenness relative to degree
#'
#' Brokers sit on many shortest paths without having many direct ties. The
#' score is the ratio of betweenness percentile to degree percentile.
#'
#' @param g An `igraph` object.
#' @param threshold Minimum broker score for a vertex to be flagged.
#' @param weighted Logical; use the `weight` edge attribute when present.
#' @return A tibble with `name`, `degree`, `betweenness`, `broker_score` and
#'   logical `broker`, sorted by decreasing score.
#' @export
#' @family roles
flag_brokers <- function(g, threshold = 1.5, weighted = TRUE) {
  centrality_table(g, c("degree", "betweenness"), weighted = weighted) %>%
    dplyr::mutate(
      deg_pct = dplyr::percent_rank(degree),
      btw_pct = dplyr::percent_rank(betweenness),
      broker_score = dplyr::if_else(deg_pct > 0, btw_pct / deg_pct, NA_real_),
      broker = !is.na(broker_score) & broker_score >= threshold
    ) %>%
    dplyr::select(name, degree, betweenness, broker_score, broker) %>%
    dplyr::arrange(dplyr::desc(broker_score))
}

#' Flag hubs: high degree and high eigenvector centrality
#'
#' Hubs are well connected and connected to other well-connected vertices.
#' The score is the mean of the degree and eigenvector percentiles.
#'
#' @param g An `igraph` object.
#' @param threshold Minimum hub score for a vertex to be flagged.
#' @param weighted Logical; use the `weight` edge attribute when present.
#' @return A tibble with `name`, `degree`, `eigen`, `hub_score` and logical
#'   `hub`, sorted by decreasing score.
#' @export
#' @family roles
flag_hubs <- function(g, threshold = 0.9, weighted = TRUE) {
  centrality_table(g, c("degree", "eigen"), weighted = weighted) %>%
    dplyr::mutate(
      deg_pct = dplyr::percent_rank(degree),
      eig_pct = dplyr::percent_rank(eigen),
      hub_score = (deg_pct + eig_pct) / 2,
      hub = hub_score >= threshold
    ) %>%
    dplyr::select(name, degree, eigen, hub_score, hub) %>%
    dplyr::arrange(dplyr::desc(hub_score))
}
