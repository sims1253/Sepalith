#' @title Detect communities with a named algorithm
#'
#' @description Wraps several igraph clustering routines behind one interface
#'   and handles the directed/undirected mismatch some of them have.
#'
#' @param g An `igraph` object.
#' @param method Community detection algorithm.
#' @param seed Integer seed for the stochastic methods.
#' @return An igraph `communities` object.
#' @export
detect_communities <- function(g,
                               method = c("louvain", "leiden", "walktrap",
                                          "label_prop", "fast_greedy"),
                               seed = 1L) {
  method <- match.arg(method)
  w <- if ("weight" %in% igraph::edge_attr_names(g)) igraph::E(g)$weight else NULL

  # These three only accept undirected graphs; merge reciprocal edges first
  needs_undirected <- method %in% c("louvain", "leiden", "fast_greedy")
  if (needs_undirected && igraph::is_directed(g)) {
    g <- igraph::as.undirected(g, mode = "collapse", edge.attr.comb = list(weight = "sum"))
    w <- igraph::E(g)$weight
  }

  set.seed(seed)
  switch(method,
    louvain = igraph::cluster_louvain(g, weights = w),
    leiden = igraph::cluster_leiden(g, weights = w, objective_function = "modularity"),
    walktrap = igraph::cluster_walktrap(g, weights = w),
    label_prop = igraph::cluster_label_prop(g, weights = w),
    fast_greedy = igraph::cluster_fast_greedy(g, weights = w)
  )
}

#' @title Membership table for a community structure
#'
#' @param g An `igraph` object.
#' @param comm An igraph `communities` object for `g`.
#' @return A tibble with `name` and integer `community` columns.
#' @export
membership_table <- function(g, comm) {
  tibble::tibble(
    name = igraph::V(g)$name,
    community = as.integer(igraph::membership(comm))
  )
}

#' @title Summarise each community
#'
#' @description Reports size, internal edge density, total internal weight and
#'   the share of each community's edge weight that stays inside it.
#'
#' @param g An `igraph` object.
#' @param comm An igraph `communities` object for `g`.
#' @return A tibble with one row per community.
#' @export
community_summary <- function(g, comm) {
  memb <- igraph::membership(comm)
  edges <- igraph::as_data_frame(g, what = "edges") %>%
    tibble::as_tibble()
  if (!"weight" %in% names(edges)) {
    edges$weight <- 1
  }

  # Tag each edge with the communities at both ends
  edges <- edges %>%
    dplyr::mutate(
      c_from = memb[from],
      c_to = memb[to],
      internal = c_from == c_to
    )

  internal <- edges %>%
    dplyr::filter(internal) %>%
    dplyr::group_by(community = c_from) %>%
    dplyr::summarise(n_internal = dplyr::n(), w_internal = sum(weight), .groups = "drop")

  # Internal edges count once; crossing edges count for both communities
  touching <- dplyr::bind_rows(
    edges %>% dplyr::transmute(community = c_from, weight),
    edges %>% dplyr::filter(!internal) %>% dplyr::transmute(community = c_to, weight)
  ) %>%
    dplyr::group_by(community) %>%
    dplyr::summarise(w_total = sum(weight), .groups = "drop")

  tibble::tibble(community = as.integer(names(table(memb))), size = as.integer(table(memb))) %>%
    dplyr::left_join(internal, by = "community") %>%
    dplyr::left_join(touching, by = "community") %>%
    dplyr::mutate(
      n_internal = dplyr::coalesce(n_internal, 0L),
      w_internal = dplyr::coalesce(w_internal, 0),
      density = dplyr::if_else(size > 1, n_internal / (size * (size - 1) / 2), NA_real_),
      cohesion = dplyr::if_else(w_total > 0, w_internal / w_total, NA_real_)
    ) %>%
    dplyr::arrange(dplyr::desc(size))
}

#' @title Compare community methods by modularity
#'
#' @param g An `igraph` object.
#' @param methods Character vector of methods accepted by
#'   [detect_communities()].
#' @param seed Integer seed for the stochastic methods.
#' @return A tibble with `method`, `n_communities`, `largest` and
#'   `modularity`, sorted by decreasing modularity.
#' @export
compare_methods <- function(g,
                            methods = c("louvain", "walktrap", "label_prop"),
                            seed = 1L) {
  rows <- vector("list", length(methods))

  for (i in seq_along(methods)) {
    comm <- tryCatch(
      detect_communities(g, methods[i], seed = seed),
      error = function(e) {
        warning("Method ", methods[i], " failed: ", conditionMessage(e), call. = FALSE)
        NULL
      }
    )
    # A failed method just leaves a NULL slot that bind_rows() drops
    if (is.null(comm)) {
      next
    }

    sizes <- igraph::sizes(comm)
    rows[[i]] <- tibble::tibble(
      method = methods[i],
      n_communities = length(sizes),
      largest = max(sizes),
      modularity = igraph::modularity(comm)
    )
  }

  dplyr::bind_rows(rows) %>%
    dplyr::arrange(dplyr::desc(modularity))
}

#' @title Collapse communities into a quotient graph
#'
#' @description Each community becomes one vertex; edge weights between
#'   communities are summed and self-loops kept as internal weight.
#'
#' @param g An `igraph` object.
#' @param comm An igraph `communities` object for `g`.
#' @return An `igraph` object with one vertex per community and a `size`
#'   vertex attribute.
#' @export
contract_communities <- function(g, comm) {
  memb <- as.integer(igraph::membership(comm))
  if (!"weight" %in% igraph::edge_attr_names(g)) {
    g <- igraph::set_edge_attr(g, "weight", value = 1)
  }
  g <- igraph::set_vertex_attr(g, "size", value = 1)

  # Summing a unit `size` attribute gives each community's vertex count
  q <- igraph::contract(g, memb, vertex.attr.comb = list(size = "sum", "ignore"))
  q <- igraph::simplify(
    q,
    remove.multiple = TRUE,
    remove.loops = FALSE,
    edge.attr.comb = list(weight = "sum", "ignore")
  )

  igraph::set_vertex_attr(q, "name", value = as.character(seq_len(igraph::vcount(q))))
}

#' @title Mixing matrix of a vertex attribute
#'
#' @description Counts edge weight between every pair of attribute levels, a
#'   quick check for assortativity by group. Edges touching a vertex with a
#'   missing attribute value are ignored.
#'
#' @param g An `igraph` object.
#' @param attr Name of a categorical vertex attribute.
#' @param normalize Logical; divide by the total weight so cells sum to one.
#' @return A square numeric matrix indexed by attribute level.
#' @export
mixing_matrix <- function(g, attr, normalize = TRUE) {
  vals <- igraph::vertex_attr(g, attr)
  if (is.null(vals)) {
    stop("No vertex attribute named `", attr, "`.", call. = FALSE)
  }

  ends <- igraph::ends(g, igraph::E(g), names = FALSE)
  w <- igraph::E(g)$weight %||% rep(1, nrow(ends))

  # Sort on the raw values so numeric ids order as 1, 2, ..., 10
  lv <- as.character(sort(unique(vals)))
  mm <- matrix(0, length(lv), length(lv), dimnames = list(lv, lv))

  for (i in seq_len(nrow(ends))) {
    a <- as.character(vals[ends[i, 1]])
    b <- as.character(vals[ends[i, 2]])
    if (is.na(a) || is.na(b)) {
      next
    }
    mm[a, b] <- mm[a, b] + w[i]
    if (!igraph::is_directed(g) && a != b) {
      mm[b, a] <- mm[b, a] + w[i]
    }
  }

  if (normalize && sum(mm) > 0) {
    mm <- mm / sum(mm)
  }
  mm
}

#' @title Mixing matrix of community membership
#'
#' @param g An `igraph` object.
#' @param comm An igraph `communities` object for `g`.
#' @param normalize Logical; divide by the total weight so cells sum to one.
#' @return A square numeric matrix indexed by community id.
#' @export
community_mixing <- function(g, comm, normalize = TRUE) {
  g <- igraph::set_vertex_attr(g, "community", value = as.integer(igraph::membership(comm)))
  mixing_matrix(g, "community", normalize = normalize)
}
