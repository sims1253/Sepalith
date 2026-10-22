# skipped: needs future/parallelly (plan_workers, with_workers),
#   furrr (par_map_furrr); par_map() is only checked on its sequential path

# detect_workers()
Sys.setenv(TASKRELAY_WORKERS = "3")
stopifnot(identical(detect_workers(), 3L))
Sys.setenv(TASKRELAY_WORKERS = "0")
stopifnot(identical(detect_workers(), 1L))
Sys.setenv(TASKRELAY_WORKERS = "lots")
w <- withCallingHandlers(detect_workers(max_workers = 2L),
                         warning = function(w) invokeRestart("muffleWarning"))
Sys.unsetenv("TASKRELAY_WORKERS")
stopifnot(is.integer(w), w >= 1L, w <= 2L, identical(detect_workers(max_workers = 1), 1L))

# chunk_indices()
ch <- chunk_indices(10, 3)
stopifnot(
  length(ch) == 3L,
  identical(unlist(ch), 1:10),
  identical(chunk_indices(0, 4), list()),
  length(chunk_indices(2, 5)) == 2L
)

# par_map() and collect_errors()
res <- par_map(list(a = 1, b = "x", c = 3), function(v, k) v * k, backend = "sequential",
               workers = 1L, k = 2)
errs <- collect_errors(res)
stopifnot(
  identical(res$a, 2), identical(res$c, 6),
  inherits(res$b, "error"),
  identical(names(errs), "b"),
  identical(par_map(list(), identity), list()),
  identical(collect_errors(list(1, 2)), character()),
  identical(names(collect_errors(list(1, simpleError("e")))), "2")
)
