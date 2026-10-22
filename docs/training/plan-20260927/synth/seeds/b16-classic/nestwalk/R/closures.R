#' Make a counter
#'
#' Returns a set of closures sharing one private count, so several callers
#' can increment and read the same counter.
#'
#' @param start Initial value of the count.
#' @param step Amount added by each call to `increment()`.
#' @return A list of functions `increment()`, `get()` and `reset()`.
#' @export
make_counter <- function(start = 0, step = 1) {
  count <- start
  increment <- function() {
    count <<- count + step
    return(invisible(count))
  }
  get <- function() {
    return(count)
  }
  reset <- function() {
    count <<- start
    return(invisible(count))
  }
  return(list(increment = increment, get = get, reset = reset))
}

#' Make a running accumulator
#'
#' Each call adds its argument to a running total and returns the new total.
#'
#' @param init Starting total.
#' @return A function of one numeric argument.
#' @export
make_accumulator <- function(init = 0) {
  total <- init
  return(function(x) {
    total <<- total + sum(x)
    return(total)
  })
}

#' Make a function that adds a constant
#'
#' @param n Number to add.
#' @return A function of one argument `x` returning `x + n`.
#' @family function factories
#' @export
make_adder <- function(n) {
  force(n)
  return(function(x) {
    return(x + n)
  })
}

#' Make a function that multiplies by a constant
#'
#' @param n Number to multiply by.
#' @return A function of one argument `x` returning `x * n`.
#' @family function factories
#' @export
make_multiplier <- function(n) {
  force(n)
  return(function(x) {
    return(x * n)
  })
}

#' Make a function that raises to a power
#'
#' @param n Exponent to apply.
#' @return A function of one argument `x` returning `x ^ n`.
#' @family function factories
#' @export
make_power <- function(n) {
  force(n)
  return(function(x) {
    return(x^n)
  })
}

#' Compose any number of functions
#'
#' Functions are applied right to left, so `compose_all(f, g)(x)` equals
#' `f(g(x))`.
#'
#' @param ... Functions of one argument.
#' @return A function of one argument.
#' @export
compose_all <- function(...) {
  fns <- list(...)
  if (length(fns) == 0L) {
    return(identity)
  }
  if (!all(vapply(fns, is.function, logical(1)))) {
    stop("all arguments to compose_all() must be functions", call. = FALSE)
  }
  return(function(x) {
    for (f in rev(fns)) {
      x <- f(x)
    }
    return(x)
  })
}

#' Limit how often a function can run
#'
#' Wraps `f` so that it runs at most `max_calls` times. Further calls return
#' `fallback` without calling `f`.
#'
#' @param f Function to wrap.
#' @param max_calls Maximum number of calls passed through to `f`.
#' @param fallback Value returned once the limit is reached.
#' @return A function with the same arguments as `f`.
#' @export
make_call_limiter <- function(f, max_calls = 1L, fallback = NULL) {
  force(f)
  calls <- 0L
  return(function(...) {
    if (calls >= max_calls) {
      return(fallback)
    }
    calls <<- calls + 1L
    return(f(...))
  })
}

#' Memoise a function of one argument
#'
#' Results are stored in a private environment keyed by the deparsed
#' argument, so repeated calls with the same value skip the computation.
#'
#' @param f Function of one argument.
#' @return A function of one argument with an attached cache.
#' @export
memoise_one <- function(f) {
  force(f)
  cache <- new.env(parent = emptyenv())
  memo <- function(x) {
    key <- paste(deparse(x), collapse = "")
    if (exists(key, envir = cache, inherits = FALSE)) {
      return(get(key, envir = cache, inherits = FALSE))
    }
    value <- f(x)
    assign(key, value, envir = cache)
    return(value)
  }
  attr(memo, "cache") <- cache
  return(memo)
}
