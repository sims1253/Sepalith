# skipped: parseJsonLines (needs jsonlite)

rec <- list(meta = list(page = 2L, tags = list("a", "b")), items = list(1, 2, 3), ok = TRUE)

# pluckPath
stopifnot(identical(pluckPath(rec, c("meta", "page")), 2L))
stopifnot(identical(pluckPath(rec, list("items", 3)), 3))
stopifnot(is.null(pluckPath(rec, c("meta", "missing"))))
stopifnot(identical(pluckPath(rec, list("items", 9), default = 0), 0))

# pluckChr / pluckInt / pluckLgl
stopifnot(identical(pluckChr(rec, c("meta", "page")), "2"))
stopifnot(identical(pluckInt(rec, c("meta", "page")), 2L))
stopifnot(identical(pluckLgl(rec, "ok"), TRUE))
stopifnot(is.na(pluckInt(rec, "items")), is.na(pluckChr(rec, "nope")))

# flattenRecord
flat <- flattenRecord(list(a = 1, b = list(c = "x", d = list(e = 2)), f = NULL, g = 1:3))
stopifnot(identical(names(flat), c("a", "b.c", "b.d.e", "f", "g")))
stopifnot(nrow(flat) == 1L, is.na(flat$f), identical(flat$g, "1|2|3"))
stopifnot(identical(names(flattenRecord(list(a = list(b = 1)), sep = "_")), "a_b"))

# bindRowsFill
bound <- bindRowsFill(list(data.frame(a = 1, b = "x"), data.frame(a = 2, c = TRUE)))
stopifnot(identical(names(bound), c("a", "b", "c")), nrow(bound) == 2L)
stopifnot(is.na(bound$c[1]), is.na(bound$b[2]))

# extractEvents
events <- list(
  list(user = list(id = "u1"), n = 3, ok = TRUE, at = "2026-01-02T03:04:05"),
  list(user = list(id = "u2"), ok = FALSE)
)
schema <- list(
  user = list(path = c("user", "id"), type = "chr"),
  n = list(path = "n", type = "int"),
  ok = list(path = "ok", type = "lgl"),
  at = list(path = "at", type = "time")
)
ev <- extractEvents(events, schema)
stopifnot(nrow(ev) == 2L, identical(ev$user, c("u1", "u2")), identical(ev$n, c(3L, NA)))
stopifnot(inherits(ev$at, "POSIXct"), identical(unname(attr(ev, "misses")), c(0L, 1L, 0L, 1L)))
stopifnot(nrow(extractEvents(list(), schema)) == 0L)
stopifnot(inherits(try(extractEvents(events, list(x = list(path = "a", type = "num"))),
  silent = TRUE), "try-error"))
