make_chain <- function() {
  edges <- data.frame(
    from = c("a", "b", "c", "x"),
    to = c("b", "c", "d", "y"),
    stringsAsFactors = FALSE
  )
  edges_to_adjlist(edges)
}

make_dag <- function() {
  edges <- data.frame(
    from = c("design", "design", "build", "test", "docs"),
    to = c("build", "docs", "test", "ship", "ship"),
    stringsAsFactors = FALSE
  )
  edges_to_adjlist(edges, directed = TRUE)
}

test_that("bfs visits vertices in hop order and records parents", {
  adj <- make_chain()
  walk <- bfs(adj, "b")
  expect_equal(walk$vertex[1], "b")
  expect_setequal(walk$vertex, c("a", "b", "c", "d"))
  expect_equal(walk$depth[walk$vertex == "d"], 2L)
  expect_equal(walk$parent[walk$vertex == "a"], "b")
  expect_true(is.na(walk$parent[walk$vertex == "b"]))
})

test_that("dfs visits vertices in hop order and records parents", {
  adj <- make_chain()
  walk <- dfs(adj, "b")
  expect_equal(walk$vertex[1], "b")
  expect_setequal(walk$vertex, c("a", "b", "c", "d"))
  expect_equal(walk$depth[walk$vertex == "d"], 2L)
  expect_equal(walk$parent[walk$vertex == "a"], "b")
  expect_true(is.na(walk$parent[walk$vertex == "b"]))
})

test_that("max_depth limits both traversals", {
  adj <- make_chain()
  shallow_bfs <- bfs(adj, "a", max_depth = 1)
  shallow_dfs <- dfs(adj, "a", max_depth = 1)
  expect_equal(shallow_bfs$vertex, c("a", "b"))
  expect_equal(shallow_dfs$vertex, c("a", "b"))
  expect_error(bfs(adj, "zzz"), "Unknown source")
  expect_error(dfs(adj, "zzz"), "Unknown source")
})

test_that("walk_path reconstructs the route back to the root", {
  adj <- make_chain()
  walk <- bfs(adj, "a")
  expect_equal(walk_path(walk, "d"), c("a", "b", "c", "d"))
  expect_equal(walk_path(walk, "a"), "a")
  expect_identical(walk_path(walk, "y"), character(0))
})

test_that("connected_components separates disjoint pieces", {
  adj <- make_chain()
  comp <- connected_components(adj)
  expect_equal(length(unique(comp)), 2L)
  expect_equal(comp[["a"]], comp[["d"]])
  expect_false(comp[["a"]] == comp[["x"]])
  sizes <- component_sizes(comp)
  expect_equal(sizes$size, c(4L, 2L))
  expect_equal(sum(sizes$share), 1)
})

test_that("is_bipartite detects odd cycles", {
  even <- edges_to_adjlist(data.frame(from = c("a", "b", "c", "d"), to = c("b", "c", "d", "a")))
  odd <- edges_to_adjlist(data.frame(from = c("a", "b", "c"), to = c("b", "c", "a")))
  res_even <- is_bipartite(even)
  res_odd <- is_bipartite(odd)
  expect_true(res_even$bipartite)
  expect_false(res_odd$bipartite)
  expect_equal(unname(res_even$side[c("a", "c")]), c(0L, 0L))
  expect_null(res_odd$side)
})

test_that("topological sorts agree on a small DAG", {
  adj <- make_dag()
  kahn <- topo_sort_kahn(adj)
  dfs_order <- topo_sort_dfs(adj)
  expect_equal(kahn[1], "design")
  expect_equal(dfs_order[1], "design")
  expect_equal(kahn[length(kahn)], "ship")
  expect_equal(dfs_order[length(dfs_order)], "ship")
  expect_true(is_dag(adj))
})

test_that("cpm_schedule flags the critical chain", {
  adj <- make_dag()
  duration <- c(design = 2, build = 5, docs = 1, test = 3, ship = 1)
  sched <- cpm_schedule(adj, duration)
  expect_equal(max(sched$earliest_finish), 11)
  expect_setequal(sched$task[sched$critical], c("design", "build", "test", "ship"))
  expect_equal(sched$slack[sched$task == "docs"], 7)
})
