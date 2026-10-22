# The class lives in classes.R; define a minimal stand-in here.
setClass(
  "AssayShelf",
  representation(assays = "list", rowInfo = "data.frame", colInfo = "data.frame", meta = "list")
)
obj <- new(
  "AssayShelf",
  assays = list(counts = matrix(c(0, 1, 2, NA, 4, 5), nrow = 3)),
  rowInfo = data.frame(row.names = paste0("g", 1:3)),
  colInfo = data.frame(cond = c("a", "b"), row.names = c("s1", "s2")),
  meta = list()
)

stopifnot(
  identical(.abbrev_names(NULL), "(none)"),
  identical(.abbrev_names(c("a", "b", "c")), "a b c"),
  identical(.abbrev_names(letters[1:6]), "a b ... e f"),
  identical(.format_row_line(obj), "rownames(3): g1 g2 g3"),
  identical(.format_col_line(obj), "colnames(2): s1 s2"),
  identical(.format_info_line(obj@colInfo, "colInfo"), "colInfo names(1): cond")
)

out <- capture.output(show(obj))
stopifnot(identical(out[1:2], c("class: AssayShelf", "dim: 3 2")))

s <- summary(obj)
stopifnot(
  inherits(s, "summary_AssayShelf"),
  s$assays$n_na == 1,
  s$assays$max == 5,
  isTRUE(all.equal(s$assays$pct_zero, 20))
)
printed <- capture.output(print(s))
stopifnot(any(grepl("No QC filters applied", printed)))
