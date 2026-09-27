.as_numeric_matrix <- function(x, name = "x") {
  if (is.data.frame(x)) {
    x <- data.matrix(x)
  } else {
    x <- as.matrix(x)
  }
  storage.mode(x) <- "double"
  if (!is.numeric(x) || length(dim(x)) != 2L) {
    stop(name, " must be coercible to a numeric matrix.", call. = FALSE)
  }
  if (nrow(x) < 1L || ncol(x) < 2L) {
    stop(name, " must have at least one row and two columns.", call. = FALSE)
  }
  if (anyNA(x) || any(!is.finite(x))) {
    stop(name, " must contain only finite, non-missing values.", call. = FALSE)
  }
  x
}