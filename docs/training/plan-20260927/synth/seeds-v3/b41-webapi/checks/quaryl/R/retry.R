cl <- classify_http_error(c(200L, 401L, 403L, 429L, 500L, 503L, 504L, 408L, NA, NA, 404L),
  c(rep("", 8), "Operation timed out", "could not resolve host", ""))
stopifnot(identical(cl, c("ok", "auth", "auth", "rate_limit", "server", "server", "timeout",
  "timeout", "timeout", "network", "client")))
stopifnot(identical(classify_http_error(integer(0)), character(0)))

stopifnot(backoff_delay(3) == 1, backoff_delay(3, "linear", base = 2) == 6)
stopifnot(backoff_delay(4, "exponential") == 8, backoff_delay(10, "exponential") == 60)
set.seed(1)
j <- backoff_delay(3, "jitter")
stopifnot(j >= 0, j <= 4)
stopifnot(backoff_delay(1, "fibonacci") == 1, backoff_delay(6, "fibonacci") == 8)
stopifnot(inherits(try(backoff_delay(0), silent = TRUE), "try-error"))

for (cls in c("timeout", "rate_limit", "server", "network", "auth")) {
  p <- retry_policy(cls)
  stopifnot(identical(names(p), c("retry", "max_attempts", "strategy", "base",
    "respect_retry_after")))
}
stopifnot(!retry_policy("auth")$retry, retry_policy("rate_limit")$max_attempts == 5L)
stopifnot(retry_policy("server")$strategy == "jitter", retry_policy("network")$base == 1)

e <- tryCatch(http_condition(429L, "slow down", 7), quaryl_rate_limit = function(e) e)
stopifnot(identical(e$retry_after, 7), identical(conditionMessage(e), "HTTP 429: slow down"))
e <- tryCatch(http_condition(NA, "timed out"), quaryl_timeout = function(e) e)
stopifnot(identical(e$error_class, "timeout"), identical(conditionMessage(e), "timed out"))
stopifnot(inherits(try(http_condition(200L), silent = TRUE), "try-error"))

calls <- 0L
slept <- numeric(0)
f <- function() {
  calls <<- calls + 1L
  if (calls == 1L) http_condition(503L, "busy", 3)
  if (calls == 2L) http_condition(NA, "timed out")
  if (calls == 3L) http_condition(429L, "later", 20)
  "done"
}
set.seed(2)
r <- with_retry(f, sleep = function(s) slept <<- c(slept, s))
stopifnot(identical(r$value, "done"), nrow(r$log) == 3L)
stopifnot(identical(r$log$error_class, c("server", "timeout", "rate_limit")))
stopifnot(r$log$delay[1] == 3, r$log$delay[2] == 4, r$log$delay[3] == 20)
n <- function() http_condition(NA, "connection reset")
err <- tryCatch(with_retry(n, sleep = function(s) NULL), quaryl_network = function(e) e)
stopifnot(identical(err$error_class, "network"))
stopifnot(identical(slept, r$log$delay))
g <- function() http_condition(401L, "nope")
stopifnot(inherits(try(with_retry(g, sleep = function(s) NULL), silent = TRUE), "try-error"))
h <- function() http_condition(NA, "timed out")
err <- tryCatch(with_retry(h, sleep = function(s) NULL), quaryl_timeout = function(e) e)
stopifnot(inherits(err, "quaryl_http_error"))
stopifnot(identical(with_retry(function() 1)$value, 1))

lg <- rbind(r$log, data.frame(attempt = 4L, error_class = "network", delay = 2))
s <- retry_summary(lg)
stopifnot(identical(s$error_class, c("timeout", "rate_limit", "server", "network")))
stopifnot(identical(s$n, c(1L, 1L, 1L, 1L)), identical(s$total_delay, c(4, 20, 3, 2)))
s0 <- retry_summary(data.frame(attempt = integer(0), error_class = character(0),
  delay = numeric(0)))
stopifnot(all(s0$n == 0L), all(s0$total_delay == 0))
