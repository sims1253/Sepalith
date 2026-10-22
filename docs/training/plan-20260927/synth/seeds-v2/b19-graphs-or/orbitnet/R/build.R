#' @title Read a contact log into a graph
#'
#' @description Aggregates repeated contacts between the same pair of people
#'   into a single weighted edge before building the igraph object.
#'
#' @param contacts A data frame of contacts with columns `source`, `target`
#'   and `timestamp`.
#' @param directed Logical; whether contact direction matters.
#' @param min_contacts Minimum number of contacts for a pair to become an edge.
#' @return An `igraph` object with a `weight` edge attribute.
#' @importFrom magrittr %>%
#' @export
graph_from_contacts <- function(contacts, directed = FALSE, min_contacts = 1L) {
  stopifnot(all(c("source", "target", "timestamp") %in% names(contacts)))

  # Compare ids as character: factors with different levels cannot be compared
  edges <- contacts %>%
    dplyr::mutate(source = as.character(source), target = as.character(target)) %>%
    dplyr::filter(!is.na(source), !is.na(target), source != target)

  # Undirected: put each pair in canonical order so A-B and B-A collapse
  if (!directed) {
    edges <- edges %>%
      dplyr::mutate(
        a = pmin(source, target),
        b = pmax(source, target),
        source = a,
        target = b
      ) %>%
      dplyr::select(-a, -b)
  }

  edges <- edges %>%
    dplyr::group_by(source, target) %>%
    dplyr::summarise(
      weight = dplyr::n(),
      first_seen = min(timestamp),
      last_seen = max(timestamp),
      .groups = "drop"
    ) %>%
    dplyr::filter(weight >= min_contacts)

  igraph::graph_from_data_frame(edges, directed = directed)
}

#' @title Read a transaction ledger into a graph
#'
#' @description Aggregates repeated transfers between the same pair of
#'   accounts into a single weighted edge before building the igraph object.
#'
#' @param ledger A data frame of transfers with columns `payer`, `payee` and
#'   `amount`.
#' @param directed Logical; whether transfer direction matters.
#' @param min_amount Minimum total amount for a pair to become an edge.
#' @return An `igraph` object with `weight` and `n_transfers` edge attributes.
#' @export
graph_from_ledger <- function(ledger, directed = TRUE, min_amount = 0) {
  stopifnot(all(c("payer", "payee", "amount") %in% names(ledger)))

  edges <- ledger %>%
    dplyr::mutate(payer = as.character(payer), payee = as.character(payee)) %>%
    dplyr::filter(!is.na(payer), !is.na(payee), payer != payee) %>%
    dplyr::rename(source = payer, target = payee)

  if (!directed) {
    edges <- edges %>%
      dplyr::mutate(
        a = pmin(source, target),
        b = pmax(source, target),
        source = a,
        target = b
      ) %>%
      dplyr::select(-a, -b)
  }

  # Total money moved per pair; refunds (negative amounts) net out here
  edges <- edges %>%
    dplyr::group_by(source, target) %>%
    dplyr::summarise(
      weight = sum(amount, na.rm = TRUE),
      n_transfers = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::filter(weight >= min_amount)

  igraph::graph_from_data_frame(edges, directed = directed)
}

#' @title Attach vertex attributes from a lookup table
#'
#' @description Joins a table keyed by vertex name onto the graph's vertices;
#'   vertices without a match receive `NA`.
#'
#' @param g An `igraph` object.
#' @param attrs A data frame with a `name` column plus one column per attribute.
#' @return The graph `g` with additional vertex attributes.
#' @export
add_vertex_attrs <- function(g, attrs) {
  stopifnot("name" %in% names(attrs))
  vnames <- igraph::V(g)$name
  idx <- match(vnames, as.character(attrs$name))
  for (col in setdiff(names(attrs), "name")) {
    g <- igraph::set_vertex_attr(g, col, value = attrs[[col]][idx])
  }
  g
}

#' @title Drop weak edges and the vertices they leave isolated
#'
#' @description Edges whose `weight` is missing are kept, since there is no
#'   evidence that they are weak.
#'
#' @param g An `igraph` object.
#' @param min_weight Edges with `weight` below this value are removed.
#' @param drop_isolates Logical; remove vertices of degree zero afterwards.
#' @return The pruned `igraph` object.
#' @rdname prune
#' @export
prune_by_weight <- function(g, min_weight, drop_isolates = TRUE) {
  w <- igraph::E(g)$weight
  if (is.null(w)) {
    stop("Graph has no `weight` edge attribute.", call. = FALSE)
  }

  # which() so that NA weights do not turn into NA edge indices
  g <- igraph::delete_edges(g, igraph::E(g)[which(w < min_weight)])
  if (drop_isolates) {
    g <- igraph::delete_vertices(g, igraph::V(g)[igraph::degree(g) == 0])
  }
  g
}

#' @param min_degree Vertices with degree below this value are removed.
#' @rdname prune
#' @export
prune_by_degree <- function(g, min_degree, drop_isolates = TRUE) {
  deg <- igraph::degree(g)
  g <- igraph::delete_vertices(g, igraph::V(g)[deg < min_degree])

  # Removing low-degree vertices can strand their former neighbours
  if (drop_isolates) {
    g <- igraph::delete_vertices(g, igraph::V(g)[igraph::degree(g) == 0])
  }
  g
}

#' @title Restrict a graph to its largest connected component
#'
#' @param g An `igraph` object.
#' @param mode For directed graphs, `"weak"` or `"strong"` connectivity.
#' @return The induced subgraph on the largest component.
#' @export
largest_component <- function(g, mode = c("weak", "strong")) {
  mode <- match.arg(mode)
  comp <- igraph::components(g, mode = mode)
  keep <- which(comp$membership == which.max(comp$csize))
  igraph::induced_subgraph(g, keep)
}

#' @title Graph tables as tibbles
#'
#' @description `edge_table()` returns `from`, `to` and all edge attributes;
#'   `vertex_table()` returns `name` and all vertex attributes.
#'
#' @param g An `igraph` object.
#' @return A tibble.
#' @rdname graph_tables
#' @export
edge_table <- function(g) {
  igraph::as_data_frame(g, what = "edges") %>%
    tibble::as_tibble()
}

#' @rdname graph_tables
#' @export
vertex_table <- function(g) {
  igraph::as_data_frame(g, what = "vertices") %>%
    tibble::as_tibble()
}
