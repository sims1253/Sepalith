news <- c(
  "# mypkg (development version)",
  "",
  "## Bug fixes",
  "",
  "* Fixed a crash",
  "  on empty input.",
  "",
  "# mypkg 1.10.0 (2026-03-01)",
  "",
  "* Initial bullet.",
  "",
  "## New features",
  "",
  "- Added `foo()`.",
  "",
  "# mypkg 1.9.0"
)
path <- tempfile(fileext = ".md")
writeLines(news, path)

# news_line_kind
k <- news_line_kind(news)
kinds <- c("version", "section", "bullet", "continuation", "other")
stopifnot(identical(k[c(1, 3, 5, 6, 2)], kinds))

# news_parse_heading
h <- news_parse_heading("# mypkg 1.10.0 (2026-03-01)")
stopifnot(identical(h$package, "mypkg"), identical(h$version, "1.10.0"))
stopifnot(identical(h$date, as.Date("2026-03-01")))
stopifnot(identical(news_parse_heading("# mypkg (development version)")$version, "dev"))
stopifnot(is.na(news_parse_heading("# mypkg")$version))

# news_parse / news_read
p <- news_parse(news)
stopifnot(nrow(p) == 3L, identical(p$item[1], "Fixed a crash on empty input."))
stopifnot(identical(p$version, c("dev", "1.10.0", "1.10.0")), is.na(p$section[2]))
stopifnot(identical(news_read(path), p))
stopifnot(inherits(try(news_read(tempfile()), silent = TRUE), "try-error"))

# news_versions / news_entries_for / news_latest_release
stopifnot(identical(news_versions(path), c("dev", "1.10.0", "1.9.0")))
stopifnot(nrow(news_entries_for("1.10.0", path)) == 2L)
stopifnot(inherits(try(news_entries_for("0.1", path), silent = TRUE), "try-error"))
stopifnot(identical(unique(news_latest_release(path)$version), "1.10.0"))

# news_add_bullet
news_add_bullet("Another fix.", path = path)
after <- readLines(path)
stopifnot(identical(after[5], "* Another fix."), length(after) == length(news) + 1L)
news_add_bullet("Shiny thing.", section = "New features", path = path)
after <- readLines(path)
# The dev block has no such section yet, so it is created under the heading.
stopifnot(identical(after[2:5], c("", "## New features", "", "* Shiny thing.")))
