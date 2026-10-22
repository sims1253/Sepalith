#' Depth of a nested list
#'
#' A leaf (anything that is not a list) has depth zero; an empty list has
#' depth one.
#'
#' @param x A nested list.
#' @return A single integer.
#' @export
tree_depth <- function(x) {
  if (!is.list(x)) {
    return(0L)
  }
  if (length(x) == 0L) {
    return(1L)
  }
  return(1L + max(vapply(x, tree_depth, integer(1))))
}

#' Number of leaves in a nested list
#'
#' @param x A nested list.
#' @return A single integer.
#' @export
tree_count_leaves <- function(x) {
  if (!is.list(x)) {
    return(1L)
  }
  if (length(x) == 0L) {
    return(0L)
  }
  return(sum(vapply(x, tree_count_leaves, integer(1))))
}

#' Apply a function to every leaf
#'
#' Keeps the shape and names of the list and replaces each leaf by `f(leaf)`.
#'
#' @param x A nested list.
#' @param f Function applied to each leaf.
#' @param ... Passed to `f`.
#' @return A nested list of the same shape as `x`.
#' @export
tree_map_leaves <- function(x, f, ...) {
  if (!is.list(x)) {
    return(f(x, ...))
  }
  return(lapply(x, tree_map_leaves, f = f, ...))
}

#' Flatten a nested list into paths
#'
#' Builds a path for every leaf by joining the names on the way down.
#' Unnamed elements use their position.
#'
#' @param x A nested list.
#' @param sep Separator placed between path components.
#' @param prefix Path of `x` itself; used in recursive calls.
#' @return A named list of leaves, named by path.
#' @export
tree_flatten <- function(x, sep = "/", prefix = NULL) {
  if (!is.list(x)) {
    out <- list(x)
    names(out) <- if (is.null(prefix)) "" else prefix
    return(out)
  }
  keys <- names(x)
  if (is.null(keys)) {
    keys <- rep("", length(x))
  }
  keys[keys == ""] <- as.character(which(keys == ""))
  out <- list()
  for (i in seq_along(x)) {
    path <- if (is.null(prefix)) keys[i] else paste(prefix, keys[i], sep = sep)
    out <- c(out, tree_flatten(x[[i]], sep = sep, prefix = path))
  }
  return(out)
}

#' Drop leaves that fail a test
#'
#' Removes every leaf for which `keep` returns `FALSE`. Branches left empty
#' are removed as well.
#'
#' @param x A nested list.
#' @param keep Function of one leaf returning `TRUE` or `FALSE`.
#' @return The pruned list, or `NULL` when nothing is left.
#' @export
tree_prune <- function(x, keep) {
  if (!is.list(x)) {
    if (isTRUE(keep(x))) {
      return(x)
    }
    return(NULL)
  }
  pruned <- lapply(x, tree_prune, keep = keep)
  pruned <- Filter(Negate(is.null), pruned)
  if (length(pruned) == 0L) {
    return(NULL)
  }
  return(pruned)
}

#' Find the first element with a given name
#'
#' Searches depth first and returns the first element whose name equals
#' `key`, at any level.
#'
#' @param x A nested list.
#' @param key Name to look for.
#' @return The matching element, or `NULL` if none is found.
#' @export
tree_find <- function(x, key) {
  if (!is.list(x)) {
    return(NULL)
  }
  if (!is.null(names(x)) && key %in% names(x)) {
    return(x[[key]])
  }
  for (child in x) {
    hit <- tree_find(child, key)
    if (!is.null(hit)) {
      return(hit)
    }
  }
  return(NULL)
}

#' Fibonacci numbers with a memo table
#'
#' Classic memoised recursion: results are kept in an environment so each
#' value is computed once.
#'
#' @param n A non-negative whole number.
#' @param memo Environment holding computed values; created on first call.
#' @return The `n`-th Fibonacci number as a double.
#' @export
memo_fib <- function(n, memo = new.env(parent = emptyenv())) {
  key <- as.character(n)
  if (!is.null(memo[[key]])) {
    return(memo[[key]])
  }
  value <- if (n < 2) n else memo_fib(n - 1, memo) + memo_fib(n - 2, memo)
  memo[[key]] <- value
  return(value)
}

#' Count integer partitions
#'
#' Counts the ways of writing `n` as a sum of positive parts no larger than
#' `k`, memoising on the pair `(n, k)`.
#'
#' @param n A non-negative whole number.
#' @param k Largest allowed part; defaults to `n`.
#' @param memo Environment holding computed values; created on first call.
#' @return The number of partitions as a double.
#' @export
count_partitions <- function(n, k = n, memo = new.env(parent = emptyenv())) {
  if (n == 0) {
    return(1)
  }
  if (n < 0 || k == 0) {
    return(0)
  }
  key <- paste(n, k, sep = ":")
  if (!is.null(memo[[key]])) {
    return(memo[[key]])
  }
  value <- count_partitions(n - k, k, memo) + count_partitions(n, k - 1, memo)
  memo[[key]] <- value
  return(value)
}
