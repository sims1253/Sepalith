# pipe_through
stopifnot(pipe_through(2, list(function(x) x + 1, function(x) x * 10)) == 30)
stopifnot(pipe_through("same", list()) == "same")

# running_max / running_min
stopifnot(identical(running_max(c(1, 3, 2, 5)), c(1, 3, 3, 5)))
stopifnot(identical(running_min(c(4, 2, 3, 1)), c(4, 2, 2, 1)))
stopifnot(length(running_min(numeric())) == 0)

# keep_valid
recs <- list(list(age = 30), list(age = -1), list(age = NA))
kept <- keep_valid(recs, list(function(r) r$age >= 0))
stopifnot(length(kept) == 1, kept[[1]]$age == 30)

# first_where / last_where
stopifnot(first_where(c(1, 4, 6, 3), function(v) v > 3) == 2L)
stopifnot(last_where(c(1, 4, 6, 3), function(v) v > 3) == 3L)
stopifnot(is.na(first_where(1:3, function(v) v > 9)))

# zip_with
stopifnot(identical(zip_with(function(a, b) a + b, 1:3, 4:6), c(5L, 7L, 9L)))
stopifnot(is.list(zip_with(function(a, b) c(a, b), 1:2, 3:4)))
stopifnot(inherits(try(zip_with(`+`, 1:2, 1:3), silent = TRUE), "try-error"))

# group_reduce
gr <- group_reduce(1:6, c("a", "b", "a", "b", "a", "b"), `+`, 0)
stopifnot(identical(names(gr), c("a", "b")), gr$a == 9, gr$b == 12)

# merge_all
mg <- merge_all(list(a = 1, b = list(c = 2, d = 3)), NULL, list(b = list(d = 4), e = 5))
stopifnot(identical(mg, list(a = 1, b = list(c = 2, d = 4), e = 5)))
stopifnot(identical(merge_all(), list()))
