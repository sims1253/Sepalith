now = as.POSIXct("2024-05-01 12:00:00", tz = "UTC")
s = parseRateHeaders(c("x-ratelimit-limit" = "60", "x-ratelimit-remaining" = "10",
  "x-ratelimit-reset" = format(as.numeric(now) + 30)), "standard", now)
stopifnot(s$limit == 60, s$remaining == 10, s$resetAt == now + 30)
d = parseRateHeaders(c("ratelimit-limit" = "100", "ratelimit-remaining" = "0",
  "ratelimit-reset" = "15"), "draft", now)
stopifnot(d$limit == 100, d$remaining == 0, d$resetAt == now + 15)
sh = parseRateHeaders(c("x-shopify-shop-api-call-limit" = "32/40"), "shopify", now)
stopifnot(sh$limit == 40, sh$remaining == 8, sh$resetAt == now + 1)
stopifnot(is.na(parseRateHeaders(character(0), "shopify", now)$limit))
ra = parseRateHeaders(c("retry-after" = "7"), "retryafter", now)
stopifnot(is.na(ra$limit), ra$remaining == 0, ra$resetAt == now + 7)
stopifnot(is.na(parseRateHeaders(character(0), "standard", now)$remaining))

st = updateRateState(NULL, s, now)
stopifnot(st$limit == 60, st$remaining == 10, st$resetAt == now + 30, st$requests == 1L)
stopifnot(st$lastSeen == now)
none = list(limit = NA_real_, remaining = NA_real_, resetAt = as.POSIXct(NA))
st2 = updateRateState(st, none, now + 5)
stopifnot(st2$remaining == 9, st2$requests == 2L, st2$limit == 60, st2$lastSeen == now + 5)
st3 = updateRateState(st2, none, now + 40)
stopifnot(st3$requests == 1L, st3$resetAt == now + 30)

stopifnot(rateLimitWait(st, now) == 0, rateLimitWait(st, now, "spread") == 3)
stopifnot(rateLimitWait(st, now, "reserve", reserve = 10) == 30)
stopifnot(rateLimitWait(st, now, "reserve") == 0, rateLimitWait(st, now, "none") == 0)
empty = st
empty$remaining = 0
stopifnot(rateLimitWait(empty, now) == 30, rateLimitWait(empty, now, "spread") == 30)
stopifnot(rateLimitWait(updateRateState(NULL, none, now), now) == 0)

b = list(tokens = 1, capacity = 5, rate = 2, updatedAt = now)
r1 = takeTokens(b, now, cost = 1)
stopifnot(r1$allowed, r1$bucket$tokens == 0, r1$wait == 0)
r2 = takeTokens(r1$bucket, now, cost = 3)
stopifnot(!r2$allowed, r2$wait == 1.5)
r3 = takeTokens(r1$bucket, now + 10, cost = 3)
stopifnot(r3$allowed, r3$bucket$tokens == 2, r3$bucket$updatedAt == now + 10)
stopifnot(inherits(try(takeTokens(b, now, cost = 6), silent = TRUE), "try-error"))

rep1 = rateLimitReport(list(a = st, b = empty, c = updateRateState(NULL, none, now)), now)
stopifnot(identical(names(rep1), c("endpoint", "limit", "remaining", "usedPct", "resetIn",
  "requests")))
stopifnot(identical(rep1$endpoint, c("b", "a", "c")), rep1$usedPct[1:2] == c(100, 83.3))
stopifnot(rep1$resetIn[1] == 30, is.na(rep1$resetIn[3]), is.na(rep1$usedPct[3]))
stopifnot(identical(rep1$requests, c(1L, 1L, 1L)))
