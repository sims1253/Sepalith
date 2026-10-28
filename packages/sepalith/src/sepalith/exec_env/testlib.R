# Assertion helpers for exec-env test files (convention exec-test-v1).
#
# Every assertion names a behaviour category (a short label such as "basic", "edge",
# "validation", "na"). Rewards average the pass share per category first, so many
# duplicated easy assertions cannot outweigh one hard behaviour. Each assertion is
# evaluated in isolation: an error inside it counts as a failed assertion and the
# script continues. Only these helpers count; printing and exit status never do.
#
#   check(cat, expr)                         expr must be exactly TRUE
#   check_equal(cat, actual, expected, tol)  all.equal with tolerance (default 1e-8)
#   check_identical(cat, actual, expected)   identical()
#   check_error(cat, expr, class=, pattern=) expr must signal an error of that class
#                                            and/or with a message matching pattern
#   check_warning(cat, expr, pattern=)       expr must signal a matching warning
#
# strict(x, class = NULL): a fake object (list) whose `$` and `[[` raise an error for
# any field it does not have, so code that reads fields the real object would carry
# cannot pass silently on NULL. Its class is c(class, "exec_strict").
.exec_state <- new.env(parent = emptyenv())
.exec_state$category <- character()
.exec_state$ok <- logical()

.exec_record <- function(category, ok) {
  if (!is.character(category) || length(category) != 1L || !nzchar(category)) {
    stop("an assertion needs a single category label", call. = FALSE)
  }
  n <- length(.exec_state$ok) + 1L
  .exec_state$category[n] <- category
  .exec_state$ok[n] <- isTRUE(ok)
  invisible(isTRUE(ok))
}

.exec_timeout <- function(e) grepl("reached (elapsed|CPU) time limit", conditionMessage(e))

.exec_eval <- function(thunk) {
  tryCatch(suppressWarnings(thunk()), error = function(e) {
    if (.exec_timeout(e)) stop(e)
    structure(list(e), class = ".exec_failed")
  })
}

check <- function(category, expr) {
  value <- .exec_eval(function() expr)
  .exec_record(category, !inherits(value, ".exec_failed") && isTRUE(value))
}

check_true <- check

check_equal <- function(category, actual, expected, tolerance = 1e-8) {
  value <- .exec_eval(function() isTRUE(all.equal(actual, expected, tolerance = tolerance)))
  .exec_record(category, !inherits(value, ".exec_failed") && isTRUE(value))
}

check_identical <- function(category, actual, expected) {
  value <- .exec_eval(function() identical(actual, expected))
  .exec_record(category, !inherits(value, ".exec_failed") && isTRUE(value))
}

check_error <- function(category, expr, class = NULL, pattern = NULL) {
  cond <- tryCatch({
    suppressWarnings(expr)
    NULL
  }, error = function(e) {
    if (.exec_timeout(e)) stop(e)
    e
  })
  ok <- !is.null(cond) &&
    (is.null(class) || inherits(cond, class)) &&
    (is.null(pattern) || grepl(pattern, conditionMessage(cond)))
  .exec_record(category, ok)
}

check_warning <- function(category, expr, pattern = NULL) {
  cond <- tryCatch({
    expr
    NULL
  }, warning = function(w) w, error = function(e) {
    if (.exec_timeout(e)) stop(e)
    NULL
  })
  ok <- !is.null(cond) && (is.null(pattern) || grepl(pattern, conditionMessage(cond)))
  .exec_record(category, ok)
}

strict <- function(x, class = NULL) {
  if (!is.list(x)) stop("strict() needs a list", call. = FALSE)
  structure(x, class = unique(c(class, "exec_strict")))
}

.exec_strict_get <- function(x, i) {
  fields <- names(unclass(x))
  if (is.character(i) && length(i) == 1L && !i %in% fields) {
    stop("exec-env strict object has no field '", i, "'", call. = FALSE)
  }
  unclass(x)[[i]]
}

registerS3method("$", "exec_strict", function(x, name) .exec_strict_get(x, name), envir = baseenv())
registerS3method("[[", "exec_strict", function(x, i, ...) .exec_strict_get(x, i), envir = baseenv())

.exec_log <- function() list(category = .exec_state$category, ok = .exec_state$ok)
