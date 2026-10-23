#' Triangular solves
#'
#' Solve `L x = b` for lower triangular `L` by forward substitution, or
#' `U x = b` for upper triangular `U` by back substitution.
#'
#' @param L A lower triangular square matrix.
#' @param b A numeric vector with one element per row.
#' @family decompositions
#' @return The solution vector `x`.
#' @export
forwardSub = function(L, b) {
  n = nrow(L)
  if (length(b) != n) {
    stop("`b` must have one element per row of the matrix.", call. = FALSE)
  }
  x = numeric(n)
  for (i in seq_len(n)) {
    known = if (i > 1L) sum(L[i, 1:(i - 1L)] * x[1:(i - 1L)]) else 0
    if (L[i, i] == 0) {
      stop(sprintf("Zero pivot in row %d.", i), call. = FALSE)
    }
    x[i] = (b[i] - known) / L[i, i]
  }
  x
}

#' @rdname forwardSub
#' @param U An upper triangular square matrix.
#' @export
backSub = function(U, b) {
  n = nrow(U)
  if (length(b) != n) {
    stop("`b` must have one element per row of the matrix.", call. = FALSE)
  }
  x = numeric(n)
  for (i in rev(seq_len(n))) {
    known = if (i < n) sum(U[i, (i + 1L):n] * x[(i + 1L):n]) else 0
    if (U[i, i] == 0) {
      stop(sprintf("Zero pivot in row %d.", i), call. = FALSE)
    }
    x[i] = (b[i] - known) / U[i, i]
  }
  x
}

#' LU factorisation with partial pivoting
#'
#' Computes `P A = L U` with unit lower triangular `L`.
#'
#' @param A A square numeric matrix.
#' @return A list with elements `L`, `U` and `perm`, where `perm` gives the
#'   row order so that `A[perm, ]` equals `L %*% U`.
#' @keywords internal
luFactor = function(A) {
  n = nrow(A)
  U = A
  L = diag(n)
  perm = seq_len(n)
  for (k in seq_len(n - 1L)) {
    # Bring the largest remaining pivot into row `k`
    p = k - 1L + which.max(abs(U[k:n, k]))
    if (p != k) {
      U[c(k, p), ] = U[c(p, k), ]
      perm[c(k, p)] = perm[c(p, k)]
      if (k > 1L) {
        L[c(k, p), 1:(k - 1L)] = L[c(p, k), 1:(k - 1L)]
      }
    }
    if (U[k, k] == 0) {
      next
    }
    for (i in (k + 1L):n) {
      L[i, k] = U[i, k] / U[k, k]
      U[i, ] = U[i, ] - L[i, k] * U[k, ]
    }
  }
  list(L = L, U = U, perm = perm)
}

#' Decompose a matrix
#'
#' @param A A square numeric matrix.
#' @param method One of `"lu"`, `"qr"`, `"cholesky"`, `"eigen"` or `"svd"`.
#' @family decompositions
#' @return An object of class `mw_decomp` with elements `method`, `factors`
#'   and `dim`.
#' @export
matDecompose = function(A, method = c("lu", "qr", "cholesky", "eigen", "svd")) {
  method = match.arg(method)
  if (!is.matrix(A) || !is.numeric(A)) {
    stop("`A` must be a numeric matrix.", call. = FALSE)
  }
  if (nrow(A) != ncol(A)) {
    stop("`A` must be square.", call. = FALSE)
  }

  factors = switch(method,
    lu = {
      lu = luFactor(A)
      list(L = lu$L, U = lu$U, perm = lu$perm)
    },
    qr = {
      q = qr(A)
      list(Q = qr.Q(q), R = qr.R(q), pivot = q$pivot)
    },
    cholesky = {
      if (!isSymmetric(A)) {
        stop("`A` must be symmetric for method \"cholesky\".", call. = FALSE)
      }
      list(L = t(chol(A)))
    },
    eigen = {
      if (!isSymmetric(A)) {
        stop("`A` must be symmetric for method \"eigen\".", call. = FALSE)
      }
      e = eigen(A, symmetric = TRUE)
      list(values = e$values, vectors = e$vectors)
    },
    svd = {
      s = svd(A)
      list(d = s$d, u = s$u, v = s$v)
    }
  )
  structure(list(method = method, factors = factors, dim = dim(A)), class = "mw_decomp")
}

#' Rebuild a matrix from its decomposition
#'
#' @param dec An object of class `mw_decomp` from [matDecompose()].
#' @family decompositions
#' @return The reconstructed square matrix.
#' @export
reconstruct = function(dec) {
  stopifnot(inherits(dec, "mw_decomp"))
  f = dec$factors
  switch(dec$method,
    lu = {
      A = f$L %*% f$U
      A[order(f$perm), , drop = FALSE]
    },
    qr = {
      A = f$Q %*% f$R
      A[, order(f$pivot), drop = FALSE]
    },
    cholesky = {
      f$L %*% t(f$L)
    },
    eigen = {
      f$vectors %*% diag(f$values, nrow = length(f$values)) %*% t(f$vectors)
    },
    svd = {
      f$u %*% diag(f$d, nrow = length(f$d)) %*% t(f$v)
    }
  )
}

#' Solve a linear system with a stored decomposition
#'
#' @param dec An object of class `mw_decomp` from [matDecompose()].
#' @param b A numeric vector with one element per row.
#' @family decompositions
#' @return The solution vector `x` of `A x = b`.
#' @export
solveWith = function(dec, b) {
  stopifnot(inherits(dec, "mw_decomp"))
  f = dec$factors
  switch(dec$method,
    lu = {
      y = forwardSub(f$L, b[f$perm])
      backSub(f$U, y)
    },
    qr = {
      y = backSub(f$R, drop(crossprod(f$Q, b)))
      y[order(f$pivot)]
    },
    cholesky = {
      y = forwardSub(f$L, b)
      backSub(t(f$L), y)
    },
    eigen = {
      y = drop(crossprod(f$vectors, b)) / f$values
      drop(f$vectors %*% y)
    },
    svd = {
      y = drop(crossprod(f$u, b)) / f$d
      drop(f$v %*% y)
    }
  )
}

#' Check the structure of a decomposition
#'
#' @param dec An object of class `mw_decomp` from [matDecompose()].
#' @param tol Absolute tolerance used in every comparison.
#' @family decompositions
#' @return A named logical vector. Its elements depend on the method: `lu`
#'   gives `unitLower` and `upper`; `qr` gives `orthogonal` and `upper`;
#'   `cholesky` gives `lower` and `positiveDiag`; `eigen` gives `orthogonal`
#'   and `sorted`; `svd` gives `orthogonal`, `nonNegative` and `sorted`.
#' @export
decompCheck = function(dec, tol = 1e-10) {
  stopifnot(inherits(dec, "mw_decomp"))
  f = dec$factors
  I = diag(dec$dim[1])
  switch(dec$method,
    lu = {
      c(unitLower = all(abs(f$L[upper.tri(f$L)]) <= tol) && all(abs(diag(f$L) - 1) <= tol),
        upper = all(abs(f$U[lower.tri(f$U)]) <= tol))
    },
    qr = {
      c(orthogonal = max(abs(crossprod(f$Q) - I)) <= tol,
        upper = all(abs(f$R[lower.tri(f$R)]) <= tol))
    },
    cholesky = {
      c(lower = all(abs(f$L[upper.tri(f$L)]) <= tol),
        positiveDiag = all(diag(f$L) > 0))
    },
    eigen = {
      c(orthogonal = max(abs(crossprod(f$vectors) - I)) <= tol,
        sorted = !is.unsorted(rev(f$values)))
    },
    svd = {
      c(orthogonal = max(abs(crossprod(f$u) - I)) <= tol && max(abs(crossprod(f$v) - I)) <= tol,
        nonNegative = all(f$d >= 0),
        sorted = !is.unsorted(rev(f$d)))
    }
  )
}
