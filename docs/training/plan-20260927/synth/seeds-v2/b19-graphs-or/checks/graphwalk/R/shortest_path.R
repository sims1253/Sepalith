wadj <- list(
  a = c(b = 1, c = 4),
  b = c(c = 2, d = 6),
  c = c(d = 3),
  d = setNames(numeric(0), character(0)),
  e = setNames(numeric(0), character(0))
)

# dijkstra / shortest_path_to
sp <- dijkstra(wadj, "a")
stopifnot(sp$dist[["d"]] == 6, sp$dist[["c"]] == 3, is.infinite(sp$dist[["e"]]))
stopifnot(identical(shortest_path_to(sp, "d"), c("a", "b", "c", "d")))
stopifnot(identical(shortest_path_to(sp, "e"), character(0)))
stopifnot(identical(shortest_path_to(sp, "a"), "a"))

# bellman_ford agrees and handles negative weights
bf <- bellman_ford(wadj, "a")
stopifnot(identical(bf$dist, sp$dist))
neg <- wadj
neg$a <- c(b = 1, c = -1)
stopifnot(bellman_ford(neg, "a")$dist[["d"]] == 2)
cyc <- list(x = c(y = 1), y = c(x = -2))
stopifnot(inherits(try(bellman_ford(cyc, "x"), silent = TRUE), "try-error"))

# floyd_warshall
m <- matrix(Inf, 3, 3, dimnames = list(c("a", "b", "c"), c("a", "b", "c")))
m["a", "b"] <- 1
m["b", "c"] <- 2
m["c", "a"] <- NA
d <- floyd_warshall(m)
stopifnot(d["a", "c"] == 3, all(diag(d) == 0), is.infinite(d["c", "a"]))

# eccentricity / closeness / extent on an undirected path a-b-c
u <- matrix(c(0, 1, Inf, 1, 0, 1, Inf, 1, 0), 3, dimnames = list(letters[1:3], letters[1:3]))
du <- floyd_warshall(u)
stopifnot(identical(eccentricity(du), c(a = 2, b = 1, c = 2)))
clo <- closeness_from_dist(du)
stopifnot(clo[["b"]] == 1, abs(clo[["a"]] - 1 / 1.5) < 1e-12)
ext <- graph_extent(du)
stopifnot(ext$diameter == 2, ext$radius == 1, identical(ext$center, "b"))
