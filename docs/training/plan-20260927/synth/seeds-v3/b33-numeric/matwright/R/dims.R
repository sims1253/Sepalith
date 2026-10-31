#' Drop empty rows and columns
#'
#' A row or column is empty when every entry is `NA` or, with
#' `zeroIsEmpty = TRUE`, zero.
#'
#' @param A A numeric matrix.
#' @param rows Whether to drop empty rows.
#' @param cols Whether to drop empty columns.
#' @param zeroIsEmpty Whether zero entries count as empty.
#' @return The matrix without its empty rows and columns. The dropped indices
#'   are stored in the attributes `droppedRows` and `droppedCols`.
#' @export
dropEmpty = function(A, rows = TRUE, cols = TRUE, zeroIsEmpty = FALSE) {
  empty = is.na(A)
  if (zeroIsEmpty) {
    empty = empty | (!is.na(A) & A == 0)
  }
  droppedRows = integer(0)
  droppedCols = integer(0)

  # Drop empty rows
  if (rows) {
    droppedRows = which(rowSums(!empty) == 0)
    if (length(droppedRows) > 0L) {
      A = A[-droppedRows, , drop = FALSE]
      empty = empty[-droppedRows, , drop = FALSE]
    }
  }

  # Drop empty columns
  if (cols) {
    droppedCols = which(colSums(!empty) == 0)
    if (length(droppedCols) > 0L) {
      A = A[, -droppedCols, drop = FALSE]
      empty = empty[, -droppedCols, drop = FALSE]
    }
  }

  attr(A, "droppedRows") = unname(droppedRows)
  attr(A, "droppedCols") = unname(droppedCols)
  A
}

#' Pad a matrix on each side
#'
#' @param A A numeric matrix.
#' @param top,bottom Number of rows to add above and below.
#' @param left,right Number of columns to add on the left and right.
#' @param value The value used for the new cells.
#' @return The padded matrix.
#' @export
padMatrix = function(A, top = 0L, bottom = 0L, left = 0L, right = 0L, value = 0) {
  sides = c(top = top, bottom = bottom, left = left, right = right)
  for (side in names(sides)) {
    n = sides[[side]]
    if (length(n) != 1L || is.na(n) || n < 0 || n != round(n)) {
      stop(sprintf("`%s` must be a single non-negative whole number.", side), call. = FALSE)
    }
  }

  # Pad rows
  if (top > 0L) {
    A = rbind(matrix(value, top, ncol(A)), A)
  }
  if (bottom > 0L) {
    A = rbind(A, matrix(value, bottom, ncol(A)))
  }

  # Pad columns
  if (left > 0L) {
    A = cbind(matrix(value, nrow(A), left), A)
  }
  if (right > 0L) {
    A = cbind(A, matrix(value, nrow(A), right))
  }
  A
}

#' Reorder rows and columns
#'
#' @param A A matrix.
#' @param rowOrder Integer or character vector giving the new row order, or
#'   `NULL` to keep the rows.
#' @param colOrder Integer or character vector giving the new column order,
#'   or `NULL` to keep the columns.
#' @return The permuted matrix.
#' @export
permuteDims = function(A, rowOrder = NULL, colOrder = NULL) {
  # Resolve `rowOrder`
  if (is.null(rowOrder)) {
    rowOrder = seq_len(nrow(A))
  } else if (is.character(rowOrder)) {
    rowOrder = match(rowOrder, rownames(A))
  }
  if (anyNA(rowOrder) || !setequal(rowOrder, seq_len(nrow(A))) ||
      length(rowOrder) != nrow(A)) {
    stop("`rowOrder` must be a permutation of the rows of `A`.", call. = FALSE)
  }

  # Resolve `colOrder`
  if (is.null(colOrder)) {
    colOrder = seq_len(ncol(A))
  } else if (is.character(colOrder)) {
    colOrder = match(colOrder, colnames(A))
  }
  if (anyNA(colOrder) || !setequal(colOrder, seq_len(ncol(A))) ||
      length(colOrder) != ncol(A)) {
    stop("`colOrder` must be a permutation of the cols of `A`.", call. = FALSE)
  }
  A[rowOrder, colOrder, drop = FALSE]
}

#' Align two matrices on their dimnames
#'
#' Both matrices are expanded to the union of their row names and column
#' names; cells missing from a matrix are set to `fill`.
#'
#' @param A,B Numeric matrices with row and column names.
#' @param fill The value used for cells missing from a matrix.
#' @return A list with elements `A` and `B`, both with the same dimnames, and
#'   `addedRows` and `addedCols`, each a list with the names added to `A`
#'   and to `B`.
#' @export
alignDims = function(A, B, fill = NA) {
  # Validate the row names
  if (is.null(rownames(A)) || is.null(rownames(B))) {
    stop("`A` and `B` must both have row names.", call. = FALSE)
  }

  # Validate the column names
  if (is.null(colnames(A)) || is.null(colnames(B))) {
    stop("`A` and `B` must both have col names.", call. = FALSE)
  }

  # Union of the rows
  rows = union(rownames(A), rownames(B))
  addedRows = list(A = setdiff(rows, rownames(A)), B = setdiff(rows, rownames(B)))

  # Union of the columns
  cols = union(colnames(A), colnames(B))
  addedCols = list(A = setdiff(cols, colnames(A)), B = setdiff(cols, colnames(B)))

  outA = matrix(fill, length(rows), length(cols), dimnames = list(rows, cols))
  outA[rownames(A), colnames(A)] = A
  outB = matrix(fill, length(rows), length(cols), dimnames = list(rows, cols))
  outB[rownames(B), colnames(B)] = B
  list(A = outA, B = outB, addedRows = addedRows, addedCols = addedCols)
}

#' Split a matrix into blocks
#'
#' @param A A matrix.
#' @param rowSizes Integer vector of block heights summing to `nrow(A)`.
#' @param colSizes Integer vector of block widths summing to `ncol(A)`.
#' @return A list-matrix of sub-matrices with `length(rowSizes)` rows and
#'   `length(colSizes)` columns.
#' @export
blockSplit = function(A, rowSizes = nrow(A), colSizes = ncol(A)) {
  # Validate `rowSizes`
  if (any(rowSizes < 1) || sum(rowSizes) != nrow(A)) {
    stop("`rowSizes` must be positive and sum to the number of rows.", call. = FALSE)
  }

  # Validate `colSizes`
  if (any(colSizes < 1) || sum(colSizes) != ncol(A)) {
    stop("`colSizes` must be positive and sum to the number of cols.", call. = FALSE)
  }

  rowEnds = cumsum(rowSizes)
  rowStarts = rowEnds - rowSizes + 1L
  colEnds = cumsum(colSizes)
  colStarts = colEnds - colSizes + 1L

  blocks = matrix(list(), length(rowSizes), length(colSizes))
  for (i in seq_along(rowSizes)) {
    for (j in seq_along(colSizes)) {
      blocks[[i, j]] = A[rowStarts[i]:rowEnds[i], colStarts[j]:colEnds[j], drop = FALSE]
    }
  }
  blocks
}

#' Describe the rows and columns of a matrix
#'
#' @param A A matrix.
#' @return A list with elements `rows` and `cols`. Each is a list with
#'   elements `n`, `named`, `duplicated`, `allNA` and `constant` (indices
#'   whose non-missing entries are all equal).
#' @export
dimInfo = function(A) {
  isConstant = function(v) length(unique(v[!is.na(v)])) == 1L

  # Describe the rows
  rowNames = rownames(A)
  rows = list(
    n = nrow(A),
    named = !is.null(rowNames),
    duplicated = unique(rowNames[duplicated(rowNames)]),
    allNA = unname(which(rowSums(!is.na(A)) == 0)),
    constant = which(apply(A, 1L, isConstant))
  )

  # Describe the columns
  colNames = colnames(A)
  cols = list(
    n = ncol(A),
    named = !is.null(colNames),
    duplicated = unique(colNames[duplicated(colNames)]),
    allNA = unname(which(colSums(!is.na(A)) == 0)),
    constant = which(apply(A, 2L, isConstant))
  )
  list(rows = rows, cols = cols)
}
