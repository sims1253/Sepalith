m <- matrix(1:6 + 0, nrow = 3, dimnames = list(c("g1", "g2", "g3"), c("s1", "s2")))

# constructor wraps a bare matrix as "counts"
obj <- AssayShelf(m, colInfo = data.frame(cond = c("a", "b")))
stopifnot(
  is(obj, "AssayShelf"),
  identical(names(obj@assays), "counts"),
  identical(rownames(obj@rowInfo), c("g1", "g2", "g3")),
  identical(rownames(obj@colInfo), c("s1", "s2"))
)

# validity helpers
bad <- obj
bad@assays$other <- matrix(0, 2, 2)
stopifnot(
  is.null(.check_assay_dims(obj)),
  grepl("expected 3x2", .check_assay_dims(bad)),
  is.null(.check_row_info(obj)),
  is.null(.check_col_info(obj))
)
bad2 <- obj
bad2@colInfo <- data.frame(row.names = 1:3)
stopifnot(grepl("expected 2", .check_col_info(bad2)))

# annotation helpers
stopifnot(
  identical(dim(.empty_info(4)), c(4L, 0L)),
  identical(rownames(.empty_info(2, c("a", "b"))), c("a", "b")),
  identical(rownames(.as_info(data.frame(v = 1:2), 2, c("x", "y"), "colInfo")), c("x", "y")),
  inherits(try(.as_info(data.frame(v = 1:3), 2, NULL, "colInfo"), silent = TRUE), "try-error"),
  inherits(try(AssayShelf(list()), silent = TRUE), "try-error")
)
