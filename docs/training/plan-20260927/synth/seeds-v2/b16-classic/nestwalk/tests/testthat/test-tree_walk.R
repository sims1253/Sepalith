sample_tree <- function() {
  list(
    name = "root",
    settings = list(width = 80, height = 24, colours = list(fg = "black", bg = "white")),
    tags = list("a", "b", "c")
  )
}

test_that("tree_depth handles leaves, empty lists and nesting", {
  expect_equal(tree_depth(5), 0L)
  expect_equal(tree_depth(list()), 1L)
  expect_equal(tree_depth(sample_tree()), 3L)
})

test_that("tree_count_leaves counts every leaf once", {
  expect_equal(tree_count_leaves(1), 1L)
  expect_equal(tree_count_leaves(list()), 0L)
  expect_equal(tree_count_leaves(sample_tree()), 8L)
})

test_that("tree_map_leaves keeps the shape", {
  tree <- list(a = 1, b = list(c = 2, d = 3))
  out <- tree_map_leaves(tree, function(x, k) x * k, k = 10)
  expect_equal(out, list(a = 10, b = list(c = 20, d = 30)))
})

test_that("tree_flatten builds paths from names and positions", {
  out <- tree_flatten(sample_tree())

  expect_equal(out[["settings/colours/bg"]], "white")
  expect_equal(out[["tags/2"]], "b")
  expect_length(out, 8)

  # A custom separator is used between every path component
  expect_named(tree_flatten(list(x = list(y = 1)), sep = "."), "x.y")
})

test_that("tree_prune drops failing leaves and empty branches", {
  tree <- list(a = 1, b = list(c = -2, d = -3), e = list(f = 4))
  out <- tree_prune(tree, function(x) x > 0)

  expect_equal(out, list(a = 1, e = list(f = 4)))
  expect_null(tree_prune(list(x = -1), function(x) x > 0))
})

test_that("tree_find searches depth first", {
  tree <- sample_tree()
  expect_equal(tree_find(tree, "height"), 24)
  expect_equal(tree_find(tree, "fg"), "black")
  expect_null(tree_find(tree, "missing"))
})

test_that("memo_fib reuses its memo table", {
  memo <- new.env(parent = emptyenv())
  expect_equal(memo_fib(30, memo), 832040)
  expect_true(exists("29", envir = memo))

  # Base cases, each with a fresh memo table
  expect_equal(memo_fib(0), 0)
  expect_equal(memo_fib(1), 1)
})

test_that("count_partitions matches known values", {
  expect_equal(count_partitions(0), 1)
  expect_equal(count_partitions(5), 7)
  expect_equal(count_partitions(10), 42)
  expect_equal(count_partitions(10, 2), 6)
})
