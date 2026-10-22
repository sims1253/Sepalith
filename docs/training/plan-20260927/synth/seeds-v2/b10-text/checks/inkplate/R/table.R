# skipped: table.markdown, table.html, table.latex and render.table need the
# escape.* helpers from R/escape.R
# skipped: needs stringi and magrittr: truncate.cells

df <- data.frame(name = c("a", NA), qty = c(1L, 20L), price = c(1234.5, NA))

# format.cells
cells <- format.cells(df, digits = 1L, na = "-")
stopifnot(is.matrix(cells), identical(dim(cells), c(2L, 3L)))
stopifnot(identical(unname(cells[1, ]), c("a", "1", "1,234.5")), cells[2, "price"] == "-")
stopifnot(identical(dim(format.cells(df[0, ])), c(0L, 3L)))

# default.align
stopifnot(identical(default.align(df), c("l", "r", "r")))
stopifnot(identical(default.align(data.frame(f = factor("x"))), "l"))
