# skipped: needs httr2 and rvest (fetch_page), rvest (cache_read for an
# existing entry), xml2 (cache_write)
th <- new_throttle(delay = 0, jitter = 0)
stopifnot(
  inherits(th, "pagesieve_throttle"),
  identical(throttle_wait(th), 0),
  th$count == 1L,
  th$last > 0
)
throttle_reset(th)
stopifnot(th$count == 0L, th$last == -Inf)
stopifnot(inherits(try(throttle_wait(list()), silent = TRUE), "try-error"))

robots <- c(
  "# comment",
  "User-agent: *",
  "Disallow: /private",
  "",
  "User-agent: PageSieve",
  "User-agent: other",
  "Disallow: /drafts  # trailing comment",
  "Disallow:"
)
stopifnot(
  identical(parse_robots_rules(robots), "/drafts"),
  identical(parse_robots_rules(robots, agent = "curl"), "/private"),
  identical(parse_robots_rules(character()), character())
)

stopifnot(
  path_allowed("https://example.org", "/private"),
  !path_allowed("https://example.org/private/a.html", "/private"),
  path_allowed("HTTP://example.org/public?q=1", c("/private", "/drafts")),
  path_allowed("https://example.org/x", character())
)

pages <- fetch_pages(c("https://example.org/private/1", "https://example.org/private/2"),
                     throttle = th, rules = "/private")
stopifnot(
  identical(names(pages), c("https://example.org/private/1", "https://example.org/private/2")),
  identical(pages[[1]]$error, "disallowed by robots.txt"),
  identical(pages[[2]]$attempts, 0L)
)

stopifnot(is.null(cache_read("https://example.org/none", tempfile("cache"))))
