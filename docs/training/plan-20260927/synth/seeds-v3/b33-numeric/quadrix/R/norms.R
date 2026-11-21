#' Vector norms
#'
#' @param x A numeric vector.
#' @param type One of `"l1"`, `"l2"`, `"linf"` or `"lp"`.
#' @param p The exponent used when `type = "lp"`.
#' @return A single non-negative number.
#' @family norms
#' @export
vec_norm <- function(x, type = c("l2", "l1", "linf", "lp"), p = 3) {
  type <- match.arg(type)
  if (!is.numeric(x)) {
    stop(sprintf("`%s` must be numeric.", "x"), call. = FALSE)
  }
  if (type == "lp" && !(is.numeric(p) && length(p) == 1L && p >= 1)) {
    stop(sprintf("`%s` must be a single number of at least 1.", "p"), call. = FALSE)
  }
  return(switch(type,
    l1 = sum(abs(x)),
    l2 = sqrt(sum(x^2)),
    linf = max(abs(x)),
    lp = sum(abs(x)^p)^(1 / p)
  ))
}

#' Matrix norms
#'
#' @description
#' The induced 1- and infinity-norms are the largest absolute column and row
#' sums. The spectral and nuclear norms use the singular values.
#'
#' @param A A numeric matrix.
#' @param type One of `"one"`, `"inf"`, `"frobenius"`, `"max"`, `"spectral"`
#'   or `"nuclear"`.
#' @return A single non-negative number.
#' @family norms
#' @export
mat_norm <- function(A, type = c("one", "inf", "frobenius", "max", "spectral", "nuclear")) {
  type <- match.arg(type)
  if (!is.matrix(A) || !is.numeric(A)) {
    stop(sprintf("`%s` must be a numeric matrix.", "A"), call. = FALSE)
  }
  if (anyNA(A)) {
    stop(sprintf("`%s` must not contain missing values.", "A"), call. = FALSE)
  }

  switch(type,
    one = {
      sums <- colSums(abs(A))
      value <- max(sums)
    },
    inf = {
      sums <- rowSums(abs(A))
      value <- max(sums)
    },
    frobenius = {
      value <- sqrt(sum(A^2))
    },
    max = {
      value <- max(abs(A))
    },
    spectral = {
      sv <- svd(A, nu = 0L, nv = 0L)$d
      value <- max(sv)
    },
    nuclear = {
      sv <- svd(A, nu = 0L, nv = 0L)$d
      value <- sum(sv)
    }
  )
  return(value)
}

#' Condition number of a square matrix
#'
#' @param A A square numeric matrix.
#' @param type A norm type understood by [mat_norm()]; `"spectral"` uses the
#'   ratio of the extreme singular values directly.
#' @return A single number, `Inf` for a singular matrix.
#' @family norms
#' @export
cond_number <- function(A, type = c("one", "inf", "frobenius", "spectral")) {
  type <- match.arg(type)
  if (nrow(A) != ncol(A)) {
    stop(sprintf("`%s` must be a square matrix.", "A"), call. = FALSE)
  }
  if (type == "spectral") {
    sv <- svd(A, nu = 0L, nv = 0L)$d
    return(if (min(sv) == 0) Inf else max(sv) / min(sv))
  }
  inverse <- tryCatch(solve(A), error = function(e) NULL)
  if (is.null(inverse)) {
    return(Inf)
  }
  return(mat_norm(A, type) * mat_norm(inverse, type))
}

#' Report where each matrix norm is attained
#'
#' @param A A numeric matrix.
#' @return A data frame with columns `norm`, `value` and `where`, one row for
#'   each of `one`, `inf`, `max`, `frobenius`, `spectral` and `nuclear`.
#' @family norms
#' @export
norm_report <- function(A) {
  # One-norm: largest absolute column sum
  col_sums <- colSums(abs(A))
  one <- data.frame(norm = "one", value = max(col_sums),
                    where = sprintf("col %d", which.max(col_sums)))

  # Infinity-norm: largest absolute row sum
  row_sums <- rowSums(abs(A))
  inf <- data.frame(norm = "inf", value = max(row_sums),
                    where = sprintf("row %d", which.max(row_sums)))

  # Max-norm: largest absolute entry
  pos <- which(abs(A) == max(abs(A)), arr.ind = TRUE)[1, ]
  max_entry <- data.frame(norm = "max", value = max(abs(A)),
                          where = sprintf("row %d, col %d", pos[1], pos[2]))

  # Frobenius norm: all entries contribute
  frobenius <- data.frame(norm = "frobenius", value = sqrt(sum(A^2)),
                          where = "all entries")

  # Spectral norm: largest singular value
  sv <- svd(A, nu = 0L, nv = 0L)$d
  spectral <- data.frame(norm = "spectral", value = sv[1],
                         where = "singular value 1")

  # Nuclear norm: all singular values contribute
  nuclear <- data.frame(norm = "nuclear", value = sum(sv),
                        where = "all singular values")
  return(rbind(one, inf, max_entry, frobenius, spectral, nuclear))
}

#' Check the standard inequalities between matrix norms
#'
#' @param A A numeric matrix.
#' @param tol Relative slack allowed for rounding error.
#' @return A data frame with columns `bound`, `lhs`, `rhs` and `holds`.
#' @family norms
#' @export
norm_bounds <- function(A, tol = 1e-10) {
  m <- nrow(A)
  n <- ncol(A)
  r <- qr(A)$rank
  norms <- vapply(c("one", "inf", "frobenius", "max", "spectral"),
                  function(type) mat_norm(A, type), numeric(1))

  bounds <- list(
    spectral_frobenius = c(norms[["spectral"]], norms[["frobenius"]]),
    frobenius_rank = c(norms[["frobenius"]], sqrt(r) * norms[["spectral"]]),
    spectral_holder = c(norms[["spectral"]], sqrt(norms[["one"]] * norms[["inf"]])),
    max_spectral = c(norms[["max"]], norms[["spectral"]]),
    one_max = c(norms[["one"]], m * norms[["max"]]),
    inf_max = c(norms[["inf"]], n * norms[["max"]])
  )

  out <- data.frame(
    bound = names(bounds),
    lhs = vapply(bounds, `[`, numeric(1), 1L),
    rhs = vapply(bounds, `[`, numeric(1), 2L)
  )
  out$holds <- out$lhs <= out$rhs * (1 + tol)
  rownames(out) <- NULL
  return(out)
}

#' Scale the rows or columns of a matrix to unit norm
#'
#' @param A A numeric matrix.
#' @param margin Either `"rows"` or `"cols"`.
#' @param type A vector norm type understood by [vec_norm()].
#' @param tol Norms at or below `tol` count as zero and are left unscaled.
#' @return A list with elements `scaled` (the scaled matrix) and `norms` (the
#'   norm of each row or column before scaling).
#' @family norms
#' @export
normalize_margin <- function(A, margin = c("rows", "cols"), type = "l2", tol = 1e-12) {
  margin <- match.arg(margin)
  if (!is.matrix(A) || !is.numeric(A)) {
    stop(sprintf("`%s` must be a numeric matrix.", "A"), call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L || tol < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "tol"), call. = FALSE)
  }

  if (margin == "rows") {
    norms <- apply(A, 1L, vec_norm, type = type)
    zero <- norms <= tol
    if (any(zero)) {
      warning(sprintf("%d %s with zero norm left unscaled.", sum(zero), "rows"),
              call. = FALSE)
    }
    scaled <- A / ifelse(zero, 1, norms)
  } else {
    norms <- apply(A, 2L, vec_norm, type = type)
    zero <- norms <= tol
    if (any(zero)) {
      warning(sprintf("%d %s with zero norm left unscaled.", sum(zero), "cols"),
              call. = FALSE)
    }
    scaled <- sweep(A, 2L, ifelse(zero, 1, norms), "/")
  }
  dimnames(scaled) <- dimnames(A)
  return(list(scaled = scaled, norms = norms))
}

#' Pairwise distances between the rows of a matrix
#'
#' @param X A numeric matrix; each row is one point.
#' @param metric One of `"euclidean"`, `"manhattan"`, `"chebyshev"`,
#'   `"minkowski"` or `"cosine"`.
#' @param p The exponent used when `metric = "minkowski"`.
#' @return A symmetric `nrow(X)` by `nrow(X)` matrix.
#' @family norms
#' @export
row_distances <- function(X, metric = c("euclidean", "manhattan", "chebyshev",
                                        "minkowski", "cosine"), p = 3) {
  metric <- match.arg(metric)
  n <- nrow(X)
  out <- matrix(0, n, n, dimnames = list(rownames(X), rownames(X)))

  for (i in seq_len(n - 1L)) {
    for (j in (i + 1L):n) {
      xi <- X[i, ]
      xj <- X[j, ]
      switch(metric,
        euclidean = {
          d <- vec_norm(xi - xj, "l2")
        },
        manhattan = {
          d <- vec_norm(xi - xj, "l1")
        },
        chebyshev = {
          d <- vec_norm(xi - xj, "linf")
        },
        minkowski = {
          d <- vec_norm(xi - xj, "lp", p = p)
        },
        cosine = {
          d <- 1 - sum(xi * xj) / (vec_norm(xi, "l2") * vec_norm(xj, "l2"))
        }
      )
      out[i, j] <- d
      out[j, i] <- d
    }
  }
  return(out)
}
