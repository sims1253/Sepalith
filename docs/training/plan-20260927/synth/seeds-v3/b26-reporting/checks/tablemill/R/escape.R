stopifnot(identical(escapeMarkdown("a|b"), "a\\|b"))
stopifnot(identical(escapeMarkdown("*x_y*"), "\\*x\\_y\\*"))
stopifnot(identical(escapeMarkdown("`c:\\d`"), "\\`c:\\\\d\\`"))

stopifnot(identical(escapeHtml("a < b & c"), "a &lt; b &amp; c"))
stopifnot(identical(escapeHtml("\"q\" 'r' >"), "&quot;q&quot; &#39;r&#39; &gt;"))

s = "<a href=\"x\">Tom & 'Jo'</a>"
stopifnot(identical(unescapeHtml(escapeHtml(s)), s))
stopifnot(identical(unescapeHtml("&amp;lt;"), "&lt;"))

stopifnot(identical(escapeLatex("50% of $5 & #1"), "50\\% of \\$5 \\& \\#1"))
stopifnot(identical(escapeLatex("a_{b}"), "a\\_\\{b\\}"))
stopifnot(identical(escapeLatex("\\x"), "\\textbackslash{}x"))
stopifnot(identical(escapeLatex("~^"), "\\textasciitilde{}\\textasciicircum{}"))

stopifnot(identical(escapeCells("a|b\nc"), "a\\|b<br>c"))
stopifnot(identical(escapeCells(c("<b>", NA), "html"), c("&lt;b&gt;", NA)))
stopifnot(identical(escapeCells("5%\nx", "latex"), "5\\%\\newline x"))
stopifnot(identical(escapeCells("a\nb", "plain"), "a b"))

df = data.frame(`a_b` = c("x_y", "z"), n = 1:2, f = factor("p*q"), check.names = FALSE)
et = escapeTable(df, "markdown")
stopifnot(identical(names(et), c("a\\_b", "n", "f")))
stopifnot(identical(et[[1]], c("x\\_y", "z")), identical(et$n, 1:2))
stopifnot(identical(et$f, c("p\\*q", "p\\*q")))
stopifnot(identical(names(escapeTable(df, "html", names = FALSE)), names(df)))
stopifnot(inherits(try(escapeTable(list(a = 1)), silent = TRUE), "try-error"))

m = "a|b *c_d* `e\\f`"
stopifnot(identical(unescapeMarkdown(escapeMarkdown(m)), m))
stopifnot(identical(unescapeMarkdown("x\\_y"), "x_y"))

stopifnot(identical(stripMarkup("See [docs](http://x.y) for **bold** and *it*"),
                    "See docs for bold and it"))
stopifnot(identical(stripMarkup("`code`<br>a\\_b"), "code\na_b"))
stopifnot(identical(stripMarkup("<p>A &amp; <b>B</b><br/>C</p>", "html"), "A & B\nC"))
stopifnot(identical(stripMarkup("\\textbf{50\\%}\\newline x\\_y", "latex"), "50%\nx_y"))
stopifnot(identical(stripMarkup(escapeLatex("a\\b~c^d"), "latex"), "a\\b~c^d"))
