adj <- list(
  a = "b", b = c("a", "c"), c = c("b", "d"), d = "c",
  x = "y", y = "x"
)

# bfs / dfs
w <- bfs(adj, "b")
stopifnot(identical(w$vertex[1], "b"), setequal(w$vertex, c("a", "b", "c", "d")))
stopifnot(w$depth[w$vertex == "d"] == 2L, is.na(w$parent[1]))
stopifnot(identical(bfs(adj, "a", max_depth = 1)$vertex, c("a", "b")))
dw <- dfs(adj, "b")
stopifnot(identical(dw$vertex, c("b", "a", "c", "d")))
stopifnot(identical(dfs(adj, "a", max_depth = 1)$vertex, c("a", "b")))

# walk_path
stopifnot(identical(walk_path(bfs(adj, "a"), "d"), c("a", "b", "c", "d")))
stopifnot(identical(walk_path(w, "x"), character(0)))

# connected_components / component_sizes
comp <- connected_components(adj)
stopifnot(comp[["a"]] == comp[["d"]], comp[["a"]] != comp[["x"]])
sz <- component_sizes(comp)
stopifnot(identical(sz$size, c(4L, 2L)), sum(sz$share) == 1)

# is_bipartite
stopifnot(is_bipartite(adj)$bipartite)
tri <- list(a = c("b", "c"), b = c("a", "c"), c = c("a", "b"))
stopifnot(!is_bipartite(tri)$bipartite, is.null(is_bipartite(tri)$side))
