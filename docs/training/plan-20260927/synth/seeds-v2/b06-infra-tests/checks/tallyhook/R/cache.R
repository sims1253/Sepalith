# skipped: needs rlang (cache_key, memoise_tally) and cli (memoise_tally
#   errors, verbose cache_clear, disk_cache_read warnings, disk_cache_write
#   errors)
# Helpers from checks.R and zzz.R, reduced to their passing paths
check_string <- function(x, allow_null = FALSE) invisible(x)
check_number_between <- function(x, min = -Inf, max = Inf) {
  stopifnot(is.numeric(x), x >= min, x <= max)
  invisible(x)
}
tallyhook_is_verbose <- function() FALSE
cache_root <- tempfile("th-cache-")
tallyhook_cache_dir <- function(create = TRUE) {
  if (create) dir.create(cache_root, showWarnings = FALSE)
  cache_root
}

stopifnot(!cache_has("ledger-1"), is.null(cache_get("ledger-1")))
stopifnot(identical(cache_get("ledger-1", default = NA), NA))
cache_set("ledger-1", data.frame(x = 1:3))
stopifnot(cache_has("ledger-1"), identical(cache_get("ledger-1"), data.frame(x = 1:3)))

cache_set("ledger-short", 1, ttl = 0)
cache_set("ledger-long", 2, ttl = 3600)
Sys.sleep(0.01)
stopifnot(!cache_has("ledger-short"), cache_has("ledger-long"))
stopifnot(inherits(try(cache_set("bad", 1, ttl = -1), silent = TRUE), "try-error"))

cache_set("balance-a", 1)
cache_set("balance-b", 2)
stopifnot(identical(cache_clear("balance"), 2L), !cache_has("balance-a"))
stopifnot(identical(cache_clear(), 2L), length(ls(the_cache)) == 0L)

stopifnot(is.null(disk_cache_read("missing")))
path <- disk_cache_write("report-2025", list(total = 125.5))
stopifnot(file.exists(path), identical(disk_cache_read("report-2025"), list(total = 125.5)))
stopifnot(length(list.files(cache_root)) == 1L)
unlink(cache_root, recursive = TRUE)
