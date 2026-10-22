#' Run a value through a list of functions
#'
#' Applies the functions left to right with [Reduce()].
#'
#' @param x Starting value.
#' @param fns A list of functions of one argument.
#' @return The result of the last function.
#' @export
pipe_through <- function(x, fns) {
  if (length(fns) == 0L) {
    return(x)
  }
  return(Reduce(function(acc, f) f(acc), fns, x))
}

#' Running maximum
#'
#' @param x A numeric vector.
#' @return A numeric vector the same length as `x`.
#' @family running reductions
#' @export
running_max <- function(x) {
  if (length(x) == 0L) {
    return(numeric(0))
  }
  return(Reduce(max, x, accumulate = TRUE))
}

#' Running minimum
#'
#' @param x A numeric vector.
#' @return A numeric vector the same length as `x`.
#' @family running reductions
#' @export
running_min <- function(x) {
  if (length(x) == 0L) {
    return(numeric(0))
  }
  return(Reduce(min, x, accumulate = TRUE))
}

#' Keep records that pass every check
#'
#' @param records A list of records, each a named list.
#' @param checks A list of functions taking a record and returning `TRUE` or
#'   `FALSE`.
#' @return The records for which every check returned `TRUE`.
#' @export
keep_valid <- function(records, checks) {
  passes <- function(rec) {
    for (chk in checks) {
      if (!isTRUE(chk(rec))) {
        return(FALSE)
      }
    }
    return(TRUE)
  }
  return(Filter(passes, records))
}

#' Position of the first element matching a predicate
#'
#' @param x A vector or list.
#' @param pred Function of one element returning `TRUE` or `FALSE`.
#' @return An integer index, or `NA` when no element matches.
#' @export
first_where <- function(x, pred) {
  pos <- Position(pred, x, right = FALSE, nomatch = NA_integer_)
  return(as.integer(pos))
}

#' Position of the last element matching a predicate
#'
#' @param x A vector or list.
#' @param pred Function of one element returning `TRUE` or `FALSE`.
#' @return An integer index, or `NA` when no element matches.
#' @export
last_where <- function(x, pred) {
  pos <- Position(pred, x, right = TRUE, nomatch = NA_integer_)
  return(as.integer(pos))
}

#' Combine vectors element by element
#'
#' A thin wrapper around [Map()] that checks lengths first and simplifies
#' the result when every element has length one.
#'
#' @param f Function taking as many arguments as there are vectors.
#' @param ... Vectors or lists of equal length.
#' @return A list, or an atomic vector when it can be simplified.
#' @export
zip_with <- function(f, ...) {
  args <- list(...)
  lens <- lengths(args)
  if (length(unique(lens)) > 1L) {
    stop("all inputs to zip_with() must have the same length", call. = FALSE)
  }
  out <- Map(f, ...)
  names(out) <- NULL
  if (all(lengths(out) == 1L)) {
    return(unlist(out))
  }
  return(out)
}

#' Reduce within groups
#'
#' Splits `x` by `groups` and folds each piece with `f`, starting from
#' `init`.
#'
#' @param x A vector.
#' @param groups A vector or factor the same length as `x`.
#' @param f Function of two arguments, the accumulator and the next element.
#' @param init Starting value for every group.
#' @return A named list with one result per group.
#' @export
group_reduce <- function(x, groups, f, init) {
  if (length(x) != length(groups)) {
    stop("'x' and 'groups' must have the same length", call. = FALSE)
  }
  pieces <- split(x, groups)
  return(lapply(pieces, function(p) Reduce(f, p, init)))
}

#' Merge several nested lists
#'
#' Later lists win on conflicting keys; nested lists are merged recursively
#' with [utils::modifyList()].
#'
#' @param ... Named lists.
#' @return A single merged list.
#' @export
merge_all <- function(...) {
  lists <- Filter(Negate(is.null), list(...))
  if (length(lists) == 0L) {
    return(list())
  }
  return(Reduce(utils::modifyList, lists))
}
