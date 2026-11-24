#' Build a structured test matrix
#'
#' @param n The order of the matrix.
#' @param type One of `"identity"`, `"hilbert"`, `"vandermonde"`,
#'   `"toeplitz"`, `"circulant"` or `"tridiagonal"`.
#' @param x Generating values: the nodes for `"vandermonde"`, the first
#'   column for `"toeplitz"` and `"circulant"`, and the sub-, main- and
#'   super-diagonal values for `"tridiagonal"`. Ignored otherwise.
#' @return An `n` by `n` numeric matrix.
#' @export
specialMatrix = function(n, type = c("identity", "hilbert", "vandermonde", "toeplitz",
                                     "circulant", "tridiagonal"), x = NULL) {
  type = match.arg(type)
  needsX = c(vandermonde = n, toeplitz = n, circulant = n, tridiagonal = 3L)
  if (type %in% names(needsX) && length(x) != needsX[[type]]) {
    stop(sprintf("`x` must have length %d for type \"%s\".", needsX[[type]], type),
         call. = FALSE)
  }
  i = matrix(seq_len(n), n, n)
  j = t(i)

  switch(type,
    identity = {
      diag(n)
    },
    hilbert = {
      1 / (i + j - 1)
    },
    vandermonde = {
      outer(x, seq_len(n) - 1L, `^`)
    },
    toeplitz = {
      matrix(x[abs(i - j) + 1L], n, n)
    },
    circulant = {
      matrix(x[(i - j) %% n + 1L], n, n)
    },
    tridiagonal = {
      out = matrix(0, n, n)
      out[i - j == 1L] = x[1]
      out[i == j] = x[2]
      out[j - i == 1L] = x[3]
      out
    }
  )
}

#' Read or replace one diagonal band
#'
#' The band `k = 0` is the main diagonal, `k > 0` lies above it and `k < 0`
#' below it.
#'
#' @param A A matrix.
#' @param k The band offset.
#' @return For `getBand()`, the band values as a vector. For `setBand()`, the
#'   modified matrix.
#' @export
getBand = function(A, k = 0L) {
  if (k <= -nrow(A) || k >= ncol(A)) {
    stop(sprintf("Band %d lies outside a %d x %d matrix.", k, nrow(A), ncol(A)),
         call. = FALSE)
  }
  if (k >= 0L) {
    rows = seq_len(min(nrow(A), ncol(A) - k))
    cols = rows + k
  } else {
    cols = seq_len(min(ncol(A), nrow(A) + k))
    rows = cols - k
  }
  A[cbind(rows, cols)]
}

#' @rdname getBand
#' @param values Replacement values, recycled along the band.
#' @export
setBand = function(A, k = 0L, values) {
  if (k <= -nrow(A) || k >= ncol(A)) {
    stop(sprintf("Band %d lies outside a %d x %d matrix.", k, nrow(A), ncol(A)),
         call. = FALSE)
  }
  if (k >= 0L) {
    rows = seq_len(min(nrow(A), ncol(A) - k))
    cols = rows + k
  } else {
    cols = seq_len(min(ncol(A), nrow(A) + k))
    rows = cols - k
  }
  A[cbind(rows, cols)] = rep_len(values, length(rows))
  A
}

#' Finite-difference operator matrices
#'
#' Builds the matrix that applies a difference stencil to a vector of `n`
#' equally spaced samples. Rows near the ends simply drop the stencil
#' entries that fall outside the matrix.
#'
#' @param n The order of the matrix.
#' @param kind One of `"forward"`, `"backward"`, `"central"`, `"laplacian"`
#'   or `"biharmonic"`.
#' @param h The grid spacing.
#' @return An `n` by `n` numeric matrix.
#' @export
stencilMatrix = function(n, kind = c("forward", "backward", "central", "laplacian",
                                     "biharmonic"), h = 1) {
  kind = match.arg(kind)

  # Smallest order that holds the whole stencil
  minN = c(forward = 2L, backward = 2L, central = 3L, laplacian = 3L, biharmonic = 5L)
  if (n < minN[[kind]]) {
    stop(sprintf("`n` must be at least %d for kind \"%s\".", minN[[kind]], kind),
         call. = FALSE)
  }
  out = matrix(0, n, n)
  switch(kind,
    forward = {
      out = setBand(out, 0L, -1)
      out = setBand(out, 1L, 1)
      out / h
    },
    backward = {
      out = setBand(out, -1L, -1)
      out = setBand(out, 0L, 1)
      out / h
    },
    central = {
      out = setBand(out, -1L, -1 / 2)
      out = setBand(out, 1L, 1 / 2)
      out / h
    },
    laplacian = {
      out = setBand(out, -1L, 1)
      out = setBand(out, 0L, -2)
      out = setBand(out, 1L, 1)
      out / h^2
    },
    biharmonic = {
      out = setBand(out, -2L, 1)
      out = setBand(out, -1L, -4)
      out = setBand(out, 0L, 6)
      out = setBand(out, 1L, -4)
      out = setBand(out, 2L, 1)
      out / h^4
    }
  )
}

#' Make a square matrix symmetric
#'
#' @param A A square numeric matrix.
#' @param method How to combine `A[i, j]` and `A[j, i]`: `"average"`,
#'   `"upper"` (copy the upper triangle down), `"lower"` (copy the lower
#'   triangle up), `"max"` or `"min"`.
#' @return A symmetric matrix.
#' @export
symmetrize = function(A, method = c("average", "upper", "lower", "max", "min")) {
  method = match.arg(method)
  if (nrow(A) != ncol(A)) {
    stop("`A` must be square.", call. = FALSE)
  }
  At = t(A)
  switch(method,
    average = {
      (A + At) / 2
    },
    upper = {
      A[lower.tri(A)] = At[lower.tri(A)]
      A
    },
    lower = {
      A[upper.tri(A)] = At[upper.tri(A)]
      A
    },
    max = {
      pmax(A, At)
    },
    min = {
      pmin(A, At)
    }
  )
}

#' Random test matrices
#'
#' Draws from the current random number stream; call [set.seed()] first for
#' reproducible matrices.
#'
#' @param n The order of the matrix.
#' @param type One of `"gaussian"`, `"uniform"`, `"spd"` (symmetric positive
#'   definite), `"orthogonal"` or `"sparse"`.
#' @param density Expected share of non-zero entries for `"sparse"`.
#' @return An `n` by `n` numeric matrix.
#' @export
randomMatrix = function(n, type = c("gaussian", "uniform", "spd", "orthogonal", "sparse"),
                        density = 0.2) {
  type = match.arg(type)
  if (length(n) != 1L || is.na(n) || n < 1) {
    stop("`n` must be a single positive whole number.", call. = FALSE)
  }
  if (length(density) != 1L || is.na(density) || density < 0 || density > 1) {
    stop("`density` must be a single number between 0 and 1.", call. = FALSE)
  }
  switch(type,
    gaussian = {
      matrix(stats::rnorm(n * n), n, n)
    },
    uniform = {
      matrix(stats::runif(n * n, -1, 1), n, n)
    },
    spd = {
      G = matrix(stats::rnorm(n * n), n, n)
      crossprod(G) + n * diag(n)
    },
    orthogonal = {
      G = matrix(stats::rnorm(n * n), n, n)
      q = qr(G)
      qr.Q(q) %*% diag(sign(diag(qr.R(q))), n)
    },
    sparse = {
      G = matrix(stats::rnorm(n * n), n, n)
      G[stats::runif(n * n) > density] = 0
      G
    }
  )
}

#' Block-diagonal matrix
#'
#' @param ... Numeric matrices placed along the diagonal in order.
#' @param fill The value used off the blocks.
#' @return A matrix with `sum(nrow)` rows and `sum(ncol)` columns.
#' @export
blockDiag = function(..., fill = 0) {
  blocks = lapply(list(...), as.matrix)
  heights = vapply(blocks, nrow, integer(1))
  widths = vapply(blocks, ncol, integer(1))
  out = matrix(fill, sum(heights), sum(widths))
  rowOffset = 0L
  colOffset = 0L
  for (b in seq_along(blocks)) {
    rows = rowOffset + seq_len(heights[b])
    cols = colOffset + seq_len(widths[b])
    out[rows, cols] = blocks[[b]]
    rowOffset = rowOffset + heights[b]
    colOffset = colOffset + widths[b]
  }
  out
}
