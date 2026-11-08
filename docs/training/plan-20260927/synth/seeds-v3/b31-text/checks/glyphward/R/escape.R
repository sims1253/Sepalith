stopifnot(identical(literal.alternation(c("a", "a.b", "a")), "a\\.b|a"))
stopifnot(grepl(literal.alternation(c("$", "{")), "x{", perl = TRUE))
stopifnot(inherits(tryCatch(literal.alternation(character()), error = identity), "error"))

stopifnot(identical(map.matches(c("ab", NA, "b"), "a|b", c(a = "1", b = "2")), c("12", NA, "2")))
stopifnot(identical(map.matches("xyz", "q", c(q = "Q")), "xyz"))

stopifnot(identical(escape.table("html")[["<"]], "&lt;"))
stopifnot(identical(escape.table("latex")[["%"]], "\\%"))
stopifnot(identical(escape.table("json")[["\n"]], "\\n"))
stopifnot(identical(escape.table("markdown")[["*"]], "\\*"))

h <- "Tom & \"Jerry\" <b>"
stopifnot(identical(escape.text(h), "Tom &amp; &quot;Jerry&quot; &lt;b&gt;"))
stopifnot(identical(escape.text(h, minimal = TRUE), "Tom &amp; \"Jerry\" &lt;b&gt;"))
stopifnot(identical(unescape.text(escape.text(h)), h))
stopifnot(identical(unescape.text("&amp;lt;"), "&lt;"))
stopifnot(identical(unescape.text("&quot;", minimal = TRUE), "&quot;"))

l <- "50% of $x_1 & {y} \\ ~"
el <- escape.text(l, "latex")
stopifnot(identical(el, "50\\% of \\$x\\_1 \\& \\{y\\} \\textbackslash{} \\textasciitilde{}"))
stopifnot(identical(unescape.text(el, "latex"), l))
stopifnot(identical(escape.text("a~b^c", "latex", minimal = TRUE), "a~b^c"))

j <- "say \"hi\"\n\tpath a/b \\ end"
ej <- escape.text(j, "json", minimal = TRUE)
stopifnot(identical(ej, "say \\\"hi\\\"\\n\\tpath a/b \\\\ end"))
stopifnot(identical(unescape.text(ej, "json"), j))
stopifnot(identical(escape.text("a/b", "json"), "a\\/b"))

stopifnot(identical(escape.text(c(k = "*a* | [b]"), "markdown"), c(k = "\\*a\\* \\| \\[b\\]")))
stopifnot(identical(escape.text("a|b", "markdown", minimal = TRUE), "a|b"))
stopifnot(identical(unescape.text("\\_x\\_", "markdown"), "_x_"))

rt <- escape.roundtrip(c("a <b> & 'c'", "plain", NA))
stopifnot(all(rt$roundtrip), identical(rt$format, c("html", "latex", "json", "markdown")))
stopifnot(identical(rt$n_changed, c(1L, 1L, 0L, 0L)))
