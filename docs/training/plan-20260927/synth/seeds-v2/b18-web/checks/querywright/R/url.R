p <- url_parts("HTTPS://Ex.com:8080/a/b?x=1#top")
stopifnot(
  identical(p$scheme, "https"),
  identical(p$host, "ex.com"),
  identical(p$port, "8080"),
  identical(p$query, "x=1"),
  identical(url_parts("/rel/path")$host, "")
)

stopifnot(
  identical(url_assemble(p), "https://ex.com:8080/a/b?x=1#top"),
  identical(url_assemble(url_parts("docs/x.html")), "docs/x.html")
)

stopifnot(
  identical(url_set_path("https://ex.com/a?q=1", "b/c"), "https://ex.com/b/c?q=1"),
  identical(url_set_fragment("https://ex.com/a#x", "#y"), "https://ex.com/a#y"),
  identical(url_set_fragment("https://ex.com/a#x", ""), "https://ex.com/a"),
  identical(url_set_scheme("http://ex.com/", "HTTPS"), "https://ex.com/"),
  inherits(try(url_set_scheme("http://ex.com/", "1http"), silent = TRUE), "try-error")
)

stopifnot(
  identical(url_join("https://ex.com/api/", "v1", "/users/", 3), "https://ex.com/api/v1/users/3"),
  identical(url_join("https://ex.com", "a b"), "https://ex.com/a%20b"),
  identical(url_join("https://ex.com/x?k=1", ""), "https://ex.com/x?k=1")
)

stopifnot(
  identical(url_normalize("HTTP://Ex.com:80/a/./b/../c/#frag"), "http://ex.com/a/c"),
  identical(url_normalize("https://ex.com/../a#f", keep_fragment = TRUE), "https://ex.com/a#f"),
  identical(url_normalize("https://ex.com:8443"), "https://ex.com:8443/")
)
