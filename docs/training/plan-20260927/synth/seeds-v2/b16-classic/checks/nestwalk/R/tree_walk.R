tree <- list(
  name = "root",
  settings = list(width = 80, height = 24, colours = list(fg = "black", bg = "white")),
  tags = list("a", "b", "c")
)

# tree_depth / tree_count_leaves
stopifnot(tree_depth(5) == 0L, tree_depth(list()) == 1L, tree_depth(tree) == 3L)
stopifnot(tree_count_leaves(1) == 1L)
stopifnot(tree_count_leaves(list()) == 0L)
stopifnot(tree_count_leaves(tree) == 8L)

# tree_map_leaves
mapped <- tree_map_leaves(list(a = 1, b = list(c = 2)), function(x, k) x * k, k = 10)
stopifnot(identical(mapped, list(a = 10, b = list(c = 20))))

# tree_flatten
flat <- tree_flatten(tree)
stopifnot(length(flat) == 8, flat[["settings/colours/bg"]] == "white", flat[["tags/2"]] == "b")
stopifnot(identical(names(tree_flatten(list(x = list(y = 1)), sep = ".")), "x.y"))

# tree_prune
pr <- tree_prune(list(a = 1, b = list(c = -2, d = -3), e = list(f = 4)), function(x) x > 0)
stopifnot(identical(pr, list(a = 1, e = list(f = 4))))
stopifnot(is.null(tree_prune(list(x = -1), function(x) x > 0)))

# tree_find
stopifnot(tree_find(tree, "height") == 24)
stopifnot(tree_find(tree, "fg") == "black")
stopifnot(is.null(tree_find(tree, "nope")))

# memo_fib
memo <- new.env(parent = emptyenv())
stopifnot(memo_fib(30, memo) == 832040, exists("29", envir = memo), memo_fib(0) == 0)
stopifnot(inherits(try(memo_fib(-3), silent = TRUE), "try-error"))

# count_partitions
stopifnot(count_partitions(0) == 1, count_partitions(5) == 7, count_partitions(10) == 42)
stopifnot(count_partitions(10, 2) == 6)
