#' Rotate a matrix by a quarter turn
#'
#' @param m A matrix.
#' @return The matrix turned a quarter turn clockwise or anticlockwise.
#' @rdname rotate
#' @export
rotate.clockwise = function(m) {
  t(m[nrow(m):1, , drop = FALSE])
}

#' @rdname rotate
#' @export
rotate.anticlockwise = function(m) {
  t(m)[ncol(m):1, , drop = FALSE]
}

#' Read a matrix in spiral order
#'
#' Walks the outer ring clockwise from the top left, then the next ring in,
#' and so on.
#'
#' @param m A matrix.
#' @return A vector of the elements of `m` in spiral order.
#' @export
spiral.order = function(m) {
  # typeof() rather than mode(): mode() reports integers as "numeric"
  out = vector(typeof(m), length(m))
  k = 0L
  top = 1L
  bottom = nrow(m)
  left = 1L
  right = ncol(m)

  while (top <= bottom && left <= right) {
    # top row, left to right
    for (j in left:right) {
      k = k + 1L
      out[k] = m[top, j]
    }
    top = top + 1L

    # right column, downwards
    if (top <= bottom) {
      for (i in top:bottom) {
        k = k + 1L
        out[k] = m[i, right]
      }
    }
    right = right - 1L

    # bottom row and left column only exist if the ring is not a single line
    if (top <= bottom && left <= right) {
      for (j in right:left) {
        k = k + 1L
        out[k] = m[bottom, j]
      }
      bottom = bottom - 1L
    }
    if (left <= right && top <= bottom) {
      for (i in bottom:top) {
        k = k + 1L
        out[k] = m[i, left]
      }
      left = left + 1L
    }
  }
  out
}

#' Check whether a matrix is a magic square
#'
#' All rows, columns and both diagonals must share the same sum.
#'
#' @param m A square numeric matrix.
#' @return `TRUE` or `FALSE`.
#' @export
is.magic.square = function(m) {
  if (!is.matrix(m) || nrow(m) != ncol(m) || anyNA(m)) return(FALSE)
  target = sum(diag(m))
  sums = c(rowSums(m), colSums(m), sum(diag(m[, ncol(m):1, drop = FALSE])))
  all(sums == target)
}

#' Check the rows, columns or boxes of a Sudoku grid
#'
#' Empty cells, coded as `NA` or 0, are ignored.
#'
#' @param grid A 9 by 9 integer matrix.
#' @return A logical vector with one value per row, column or 3 by 3 box
#'   (boxes in row order).
#' @rdname sudoku.parts
#' @noRd
sudoku.rows.ok = function(grid) {
  apply(grid, 1, function(v) {
    v = v[!is.na(v) & v != 0]
    !anyDuplicated(v) && all(v %in% 1:9)
  })
}

#' @rdname sudoku.parts
#' @noRd
sudoku.cols.ok = function(grid) {
  apply(grid, 2, function(v) {
    v = v[!is.na(v) & v != 0]
    !anyDuplicated(v) && all(v %in% 1:9)
  })
}

#' @rdname sudoku.parts
#' @noRd
sudoku.boxes.ok = function(grid) {
  starts = c(1, 4, 7)
  ok = logical(9)
  k = 0
  for (r in starts) {
    for (cc in starts) {
      k = k + 1
      v = as.vector(grid[r:(r + 2), cc:(cc + 2)])
      v = v[!is.na(v) & v != 0]
      ok[k] = !anyDuplicated(v) && all(v %in% 1:9)
    }
  }
  ok
}

#' Validate a Sudoku grid
#'
#' @param grid A 9 by 9 integer matrix.
#' @param complete If `TRUE`, the grid must also have no empty cells.
#' @return `TRUE` if no rule is broken, `FALSE` otherwise.
#' @export
sudoku.valid = function(grid, complete = FALSE) {
  if (!is.matrix(grid) || !identical(dim(grid), c(9L, 9L))) {
    stop("'grid' must be a 9 by 9 matrix", call. = FALSE)
  }
  if (complete && any(is.na(grid) | grid == 0)) return(FALSE)

  all(sudoku.rows.ok(grid)) &&
    all(sudoku.cols.ok(grid)) &&
    all(sudoku.boxes.ok(grid))
}
