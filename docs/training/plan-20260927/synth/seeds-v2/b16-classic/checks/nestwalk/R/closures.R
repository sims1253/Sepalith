# make_counter: closures share one count
ctr <- make_counter(start = 10, step = 5)
ctr$increment()
ctr$increment()
stopifnot(ctr$get() == 20)
ctr$reset()
stopifnot(ctr$get() == 10)

# make_accumulator
acc <- make_accumulator(1)
acc(2)
stopifnot(acc(c(3, 4)) == 10)

# make_adder / make_multiplier / make_power, with forced arguments
adders <- lapply(1:3, make_adder)
stopifnot(adders[[1]](10) == 11, adders[[3]](10) == 13)
stopifnot(make_multiplier(3)(4) == 12, make_power(2)(1:3)[3] == 9)

# compose_all
f <- compose_all(make_adder(1), make_multiplier(2))
stopifnot(f(5) == 11, identical(compose_all()(7), 7))
stopifnot(inherits(try(compose_all(1), silent = TRUE), "try-error"))

# make_call_limiter
once <- make_call_limiter(function(x) x * 2, max_calls = 1, fallback = -1)
stopifnot(once(4) == 8, once(4) == -1)

# memoise_one: caches, and nearly-equal doubles get separate entries
calls <- 0
slow_id <- function(x) {
  calls <<- calls + 1
  x
}
m <- memoise_one(slow_id)
m(1:3)
m(1:3)
stopifnot(calls == 1, length(ls(attr(m, "cache"))) == 1)
stopifnot(identical(m(0.1 + 0.2), 0.1 + 0.2), identical(m(0.3), 0.3), calls == 3)
