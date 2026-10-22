m = matrix(1:6, nrow = 2)

# rotate.clockwise / rotate.anticlockwise
stopifnot(identical(rotate.clockwise(m), matrix(c(2L, 4L, 6L, 1L, 3L, 5L), nrow = 3)))
stopifnot(identical(rotate.anticlockwise(rotate.clockwise(m)), m))
three = rotate.clockwise(rotate.clockwise(rotate.clockwise(m)))
stopifnot(identical(three, rotate.anticlockwise(m)))

# spiral.order
sq = matrix(1:9, 3, byrow = TRUE)
stopifnot(identical(spiral.order(sq), c(1L, 2L, 3L, 6L, 9L, 8L, 7L, 4L, 5L)))
stopifnot(identical(spiral.order(m), c(1L, 3L, 5L, 6L, 4L, 2L)))
stopifnot(identical(spiral.order(matrix(letters[1:3], 3)), c("a", "b", "c")))

# is.magic.square
lo = matrix(c(2, 7, 6, 9, 5, 1, 4, 3, 8), 3)
stopifnot(is.magic.square(lo), !is.magic.square(m), !is.magic.square(matrix(1:4, 2)))
lo_na = lo
lo_na[1, 1] = NA
stopifnot(identical(is.magic.square(lo_na), FALSE))

# sudoku helpers and sudoku.valid
base = c(1, 2, 3, 4, 5, 6, 7, 8, 9)
shifts = c(0, 3, 6, 1, 4, 7, 2, 5, 8)
grid = t(sapply(shifts, function(s) (base + s - 1) %% 9 + 1))
stopifnot(all(sudoku.rows.ok(grid)), all(sudoku.cols.ok(grid)), all(sudoku.boxes.ok(grid)))
stopifnot(sudoku.valid(grid, complete = TRUE))
bad = grid
bad[1, 2] = bad[1, 1]
stopifnot(!sudoku.valid(bad), !sudoku.rows.ok(bad)[1])
holes = grid
holes[5, 5] = 0
stopifnot(sudoku.valid(holes), !sudoku.valid(holes, complete = TRUE))
stopifnot(inherits(try(sudoku.valid(matrix(0, 3, 3)), silent = TRUE), "try-error"))
