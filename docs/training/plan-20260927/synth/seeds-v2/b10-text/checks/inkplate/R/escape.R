# skipped: needs stringi and magrittr: unescape.html

# replace.by.table
stopifnot(identical(replace.by.table("a-b", c("-" = "+", "+" = "*")), "a*b"))

# escape.html / escape.attr
stopifnot(identical(escape.html("<a & b>"), "&lt;a &amp; b&gt;"))
stopifnot(identical(escape.html(c("x", NA), na = "-"), c("x", "-")))
stopifnot(identical(escape.attr("say \"hi\" 'there'"), "say &quot;hi&quot; &#39;there&#39;"))

# escape.markdown
stopifnot(identical(escape.markdown("a*b_c"), "a\\*b\\_c"))
stopifnot(identical(escape.markdown("1. item"), "1\\. item"))
stopifnot(identical(escape.markdown("- dash"), "\\- dash"))
stopifnot(identical(escape.markdown(c("1.50", "-3")), c("1.50", "-3")))

# escape.latex
stopifnot(identical(escape.latex("50% & $x_1$"), "50\\% \\& \\$x\\_1\\$"))
stopifnot(identical(escape.latex("a\\b{c}"), "a\\textbackslash{}b\\{c\\}"))
stopifnot(identical(escape.latex("~"), "\\textasciitilde{}"))

# escape.for
stopifnot(identical(escape.for("<b>"), "&lt;b&gt;"))
stopifnot(identical(escape.for("#1", "latex"), "\\#1"))
stopifnot(identical(escape.for("'", "attr"), "&#39;"))
