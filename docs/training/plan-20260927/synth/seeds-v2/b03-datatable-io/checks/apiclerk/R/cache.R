# skipped: cached_get (needs httr2 via clerk_get)

dir <- file.path(tempfile("apiclerk-"), "cache")
cache <- new_cache(dir, ttl = 5)
stopifnot(dir.exists(dir), inherits(cache, "clerk_cache"))
stopifnot(inherits(try(new_cache(dir, ttl = -1), silent = TRUE), "try-error"))

# cache_key and digest_string
k1 <- cache_key("items/list", list(page = 2, sort = "name"))
stopifnot(identical(k1, cache_key("items/list", list(sort = "name", page = 2))))
stopifnot(grepl("^items_list-[0-9a-f]{8}$", k1), nchar(digest_string("x")) == 8L)
stopifnot(identical(cache_key("items"), cache_key("items", list())))

# cache_set / cache_has / cache_get / cache_clear
stopifnot(!cache_has(cache, "k"), is.null(cache_get(cache, "k")))
cache_set(cache, "k", list(a = 1))
stopifnot(cache_has(cache, "k"), identical(cache_get(cache, "k"), list(a = 1)))
cache_set(cache, "old", 1)
Sys.setFileTime(file.path(dir, "old.rds"), Sys.time() - 60)
stopifnot(identical(cache_get(cache, "old", default = "gone"), "gone"))
stopifnot(cache_clear(cache, expired_only = TRUE) == 1L, cache_has(cache, "k"))

# print.clerk_cache
out <- capture.output(print(cache))
stopifnot(any(grepl("entries: 1", out)), any(grepl("ttl:     5 s", out)))
stopifnot(cache_clear(cache) == 1L)
unlink(dirname(dir), recursive = TRUE)
