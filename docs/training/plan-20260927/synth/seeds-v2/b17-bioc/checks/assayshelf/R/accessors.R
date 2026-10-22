# The class lives in classes.R; define a minimal stand-in here.
setClass(
  "AssayShelf",
  representation(assays = "list", rowInfo = "data.frame", colInfo = "data.frame", meta = "list")
)
m <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3)
obj <- new(
  "AssayShelf",
  assays = list(counts = m),
  rowInfo = data.frame(sym = c("A", "B", "C"), row.names = c("g1", "g2", "g3")),
  colInfo = data.frame(cond = c("x", "y"), row.names = c("s1", "s2")),
  meta = list()
)

# assay() and assay<-
stopifnot(
  identical(dimnames(assay(obj)), list(c("g1", "g2", "g3"), c("s1", "s2"))),
  is.null(dimnames(assay(obj, "counts", withDimnames = FALSE))),
  inherits(try(assay(obj, "logcounts"), silent = TRUE), "try-error")
)
assay(obj, "logcounts") <- log1p(m)
stopifnot(
  identical(names(obj@assays), c("counts", "logcounts")),
  inherits(try(assay(obj, 5) <- m, silent = TRUE), "try-error")
)

# annotation getters and setters
stopifnot(
  identical(names(rowInfo(obj, "sym")), "sym"),
  identical(colInfo(obj)$cond, c("x", "y")),
  inherits(try(colInfo(obj, "batch"), silent = TRUE), "try-error")
)
colInfo(obj) <- data.frame(batch = c(1, 2))
rowInfo(obj) <- data.frame(len = c(10, 20, 30))
stopifnot(
  identical(rownames(obj@colInfo), c("s1", "s2")),
  identical(obj@rowInfo$len, c(10, 20, 30))
)

# subsetting
sub <- obj[c("g3", "g1"), 2]
stopifnot(
  identical(dim(sub@assays$counts), c(2L, 1L)),
  identical(sub@assays$counts[, 1], c(6, 4)),
  identical(rownames(sub@rowInfo), c("g3", "g1")),
  inherits(try(obj["nope", ], silent = TRUE), "try-error")
)
