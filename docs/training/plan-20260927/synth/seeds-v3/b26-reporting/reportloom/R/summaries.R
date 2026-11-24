#' Count with a singular or plural noun
#'
#' @export
#' @param n A single count.
#' @param singular Noun used when `n` is 1.
#' @param plural Noun used otherwise.
#' @return A single string such as `"1 value"` or `"3 values"`.
pluralise <- function(n, singular, plural = paste0(singular, "s")) {
  paste(n, if (n == 1) singular else plural)
}

#' Join words into a phrase
#'
#' @export
#' @param items A character vector.
#' @param conjunction Word placed before the last item.
#' @return A single string such as `"a, b and c"`.
list.phrase <- function(items, conjunction = "and") {
  n <- length(items)
  if (n == 0L) {
    return("")
  }
  if (n == 1L) {
    return(items)
  }
  paste(paste(items[-n], collapse = ", "), conjunction, items[n])
}

#' Describe a column in one sentence
#'
#' @description
#' Methods exist for numeric, factor, logical and `Date` vectors, and for
#' data frames, which describe every column.
#'
#' @export
#' @param x The vector or data frame to describe.
#' @param name Name used as the subject of the sentence.
#' @param ... Passed on to methods.
#' @return A single string ending in a full stop; a named character vector
#'   for data frames.
describe <- function(x, name = "x", ...) {
  UseMethod("describe")
}

#' @export
#' @param digits Number of decimal places to keep.
#' @rdname describe
describe.numeric <- function(x, name = "x", digits = 1L, ...) {
  n.missing <- sum(is.na(x))
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(sprintf("%s has no observed values.", name))
  }

  low <- formatC(min(x), format = "f", digits = digits)
  high <- formatC(max(x), format = "f", digits = digits)
  centre <- formatC(mean(x), format = "f", digits = digits)
  text <- sprintf("%s ranges from %s to %s (mean %s)", name, low, high, centre)
  if (n.missing > 0L) {
    text <- paste0(text, "; ", pluralise(n.missing, "value"), " missing")
  }

  paste0(text, ".")
}

#' @export
#' @param top Number of most frequent levels to list.
#' @rdname describe
describe.factor <- function(x, name = "x", top = 3L, ...) {
  n.missing <- sum(is.na(x))
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(sprintf("%s has no observed values.", name))
  }

  counts <- sort(table(x), decreasing = TRUE)
  counts <- counts[seq_len(min(top, length(counts)))]
  levels <- sprintf("%s (%d)", names(counts), as.integer(counts))
  text <- sprintf("%s is most often %s", name, list.phrase(levels, "or"))
  if (n.missing > 0L) {
    text <- paste0(text, "; ", pluralise(n.missing, "value"), " missing")
  }

  paste0(text, ".")
}

#' @export
#' @rdname describe
describe.logical <- function(x, name = "x", digits = 1L, ...) {
  n.missing <- sum(is.na(x))
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(sprintf("%s has no observed values.", name))
  }

  hits <- sum(x)
  share <- formatC(100 * hits / length(x), format = "f", digits = digits)
  text <- sprintf("%s is true for %d of %d (%s%%)", name, hits, length(x), share)
  if (n.missing > 0L) {
    text <- paste0(text, "; ", pluralise(n.missing, "value"), " missing")
  }

  paste0(text, ".")
}

#' @export
#' @rdname describe
describe.Date <- function(x, name = "x", ...) {
  n.missing <- sum(is.na(x))
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(sprintf("%s has no observed values.", name))
  }

  first <- format(min(x), "%Y-%m-%d")
  last <- format(max(x), "%Y-%m-%d")
  span <- pluralise(as.integer(max(x) - min(x)), "day")
  text <- sprintf("%s runs from %s to %s (%s)", name, first, last, span)
  if (n.missing > 0L) {
    text <- paste0(text, "; ", pluralise(n.missing, "value"), " missing")
  }

  paste0(text, ".")
}

#' @export
#' @rdname describe
describe.data.frame <- function(x, name = "x", ...) {
  out <- character(ncol(x))
  names(out) <- names(x)
  for (column in names(x)) {
    value <- x[[column]]
    if (is.character(value)) {
      value <- factor(value)
    }
    out[[column]] <- describe(value, name = column, ...)
  }
  out
}
