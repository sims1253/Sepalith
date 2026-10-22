# skipped: needs rlang and withr (local_tallyhook_options); the error paths
#   of the getters need cli
# Passing-path stand-ins for the checkers in checks.R
check_string <- function(x, arg = "x") {
  stopifnot(is.character(x), length(x) == 1L)
  invisible(x)
}
check_flag <- function(x, arg = "x") {
  stopifnot(is.logical(x), length(x) == 1L, !is.na(x))
  invisible(x)
}
check_count <- function(x, arg = "x") invisible(as.integer(x))

defaults <- tallyhook_default_options()
stopifnot(is.list(defaults), all(startsWith(names(defaults), "tallyhook.")))
stopifnot(identical(defaults$tallyhook.max_retries, 3L))

options(tallyhook.verbose = NULL, tallyhook.currency = "CHF", tallyhook.max_retries = NULL)
.onLoad(libname = NULL, pkgname = "tallyhook")
stopifnot(isTRUE(getOption("tallyhook.verbose")), getOption("tallyhook.currency") == "CHF")
stopifnot(identical(getOption("tallyhook.max_retries"), 3L))
stopifnot(is.null(.onAttach(libname = NULL, pkgname = "tallyhook")))

options(tallyhook.currency = NULL)
stopifnot(identical(tallyhook_opt("currency"), "EUR"))
stopifnot(is.null(tallyhook_opt("nope")), identical(tallyhook_opt("nope", default = 7), 7))

options(tallyhook.verbose = FALSE, tallyhook.max_retries = 5)
stopifnot(identical(tallyhook_is_verbose(), FALSE), identical(tallyhook_max_retries(), 5L))
target <- file.path(tempfile("th-"), "nested", "cache")
options(tallyhook.cache_dir = target)
stopifnot(identical(tallyhook_cache_dir(create = FALSE), target), !dir.exists(target))
stopifnot(identical(tallyhook_cache_dir(), target), dir.exists(target))
unlink(dirname(dirname(target)), recursive = TRUE)
