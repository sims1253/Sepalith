edges <- data.frame(from = c("a", "b", "c"), to = c("b", "c", "c"))

# edges_to_adjlist
adj <- edges_to_adjlist(edges, vertices = "z")
stopifnot(identical(adj$b, c("a", "c")))
stopifnot(identical(adj$c, c("b", "c")))
stopifnot(identical(adj$z, character(0)))
dadj <- edges_to_adjlist(edges, directed = TRUE)
stopifnot(identical(dadj$a, "b"), identical(dadj$b, "c"))

# edges_to_weighted_adjlist
wedges <- data.frame(from = c("a", "b"), to = c("b", "c"), weight = c(2, 5))
wadj <- edges_to_weighted_adjlist(wedges)
stopifnot(identical(wadj$b, c(a = 2, c = 5)))
stopifnot(identical(wadj$a, c(b = 2)))
bad <- wedges
bad$weight[1] <- -1
stopifnot(inherits(try(edges_to_weighted_adjlist(bad), silent = TRUE), "try-error"))

# adjlist_to_matrix / matrix_to_adjlist round trip
m <- adjlist_to_matrix(dadj)
stopifnot(m["a", "b"] == 1L, m["b", "a"] == 0L, sum(m) == 3L)
back <- matrix_to_adjlist(m)
stopifnot(identical(back$a, "b"), identical(back$c, "c"))
m_na <- matrix(c(0, NA, 1, 0), 2, dimnames = list(c("p", "q"), c("p", "q")))
stopifnot(identical(matrix_to_adjlist(m_na)$q, character(0)))
stopifnot(identical(names(matrix_to_adjlist(diag(2))), c("1", "2")))

# out_degree / in_degree
stopifnot(identical(out_degree(dadj), c(a = 1L, b = 1L, c = 1L)))
stopifnot(identical(in_degree(dadj), c(a = 0L, b = 1L, c = 2L)))

# transpose_adjlist
tr <- transpose_adjlist(dadj)
stopifnot(identical(tr$a, character(0)), identical(tr$c, c("b", "c")))
