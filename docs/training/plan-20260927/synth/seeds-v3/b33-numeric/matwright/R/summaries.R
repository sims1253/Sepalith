#' @title One statistic per row or column
#' @description Applies a single summary statistic along one margin.
#' @param A A numeric matrix.
#' @param stat One of `"mean"`, `"median"`, `"sd"`, `"min"`, `"max"`,
#'   `"range"` or `"nNA"`.
#' @param margin Either `"rows"` or `"cols"`.
#' @param naRm Whether to drop missing values before summarising.
#' @return A numeric vector with one element per row or column.
#' @export
marginStat = function(A, stat = c("mean", "median", "sd", "min", "max", "range", "nNA"),
                      margin = c("rows", "cols"), naRm = TRUE) {
  stat = match.arg(stat)
  margin = match.arg(margin)
  fun = switch(stat,
    mean = function(v) mean(v, na.rm = naRm),
    median = function(v) stats::median(v, na.rm = naRm),
    sd = function(v) stats::sd(v, na.rm = naRm),
    min = function(v) min(v, na.rm = naRm),
    max = function(v) max(v, na.rm = naRm),
    range = function(v) diff(range(v, na.rm = naRm)),
    nNA = function(v) sum(is.na(v))
  )
  out = apply(A, if (margin == "rows") 1L else 2L, fun)
  as.numeric(out)
}

#' @title Summaries of every row or column
#' @description `rowSummary()` and `colSummary()` return the same statistics,
#'   one line per row or per column.
#' @param A A numeric matrix.
#' @param naRm Whether to drop missing values before summarising.
#' @param stats Character vector of statistics understood by [marginStat()].
#' @return A data frame with columns `index` and `name`, then one column per
#'   element of `stats`, then `share` (each row's or column's share of the
#'   matrix total).
#' @export
rowSummary = function(A, naRm = TRUE, stats = c("mean", "sd", "min", "max", "nNA")) {
  if (!is.matrix(A) || !is.numeric(A)) {
    stop("`A` must be a numeric matrix.", call. = FALSE)
  }
  unknown = setdiff(stats, c("mean", "median", "sd", "min", "max", "range", "nNA"))
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown `stats`: %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  }
  out = data.frame(index = seq_len(nrow(A)))

  # Label the rows
  out$name = if (is.null(rownames(A))) NA_character_ else rownames(A)

  # One column per statistic
  for (stat in stats) {
    out[[stat]] = marginStat(A, stat, "rows", naRm)
  }

  # Share of the total
  totals = rowSums(A, na.rm = naRm)
  out$share = totals / sum(totals)
  rownames(out) = NULL
  out
}

#' @rdname rowSummary
#' @export
colSummary = function(A, naRm = TRUE, stats = c("mean", "sd", "min", "max", "nNA")) {
  if (!is.matrix(A) || !is.numeric(A)) {
    stop("`A` must be a numeric matrix.", call. = FALSE)
  }
  unknown = setdiff(stats, c("mean", "median", "sd", "min", "max", "range", "nNA"))
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown `stats`: %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  }
  out = data.frame(index = seq_len(ncol(A)))

  # Label the columns
  out$name = if (is.null(colnames(A))) NA_character_ else colnames(A)

  # One column per statistic
  for (stat in stats) {
    out[[stat]] = marginStat(A, stat, "cols", naRm)
  }

  # Share of the total
  totals = colSums(A, na.rm = naRm)
  out$share = totals / sum(totals)
  rownames(out) = NULL
  out
}

#' @title Structural flags of a matrix
#' @description Each flag is tested up to the tolerance `tol`.
#' @param A A numeric matrix.
#' @param tol Absolute tolerance used in every comparison.
#' @return A named logical vector with elements `square`, `symmetric`,
#'   `skew`, `diagonal`, `upper`, `lower`, `tridiagonal`, `orthogonal` and
#'   `posDef`.
#' @export
structureFlags = function(A, tol = 1e-10) {
  flags = c("square", "symmetric", "skew", "diagonal", "upper", "lower", "tridiagonal",
            "orthogonal", "posDef")
  if (nrow(A) != ncol(A)) {
    return(stats::setNames(rep(FALSE, length(flags)), flags))
  }
  offset = col(A) - row(A)

  # Symmetric: equal to its transpose
  symmetric = max(abs(A - t(A))) <= tol
  # Skew: equal to minus its transpose
  skew = max(abs(A + t(A))) <= tol
  # Diagonal: no entry off the diagonal
  diagonal = all(abs(A[offset != 0]) <= tol)
  # Upper triangular: no entry below the diagonal
  upper = all(abs(A[offset < 0]) <= tol)
  # Lower triangular: no entry above the diagonal
  lower = all(abs(A[offset > 0]) <= tol)
  # Tridiagonal: no entry more than one band away from the diagonal
  tridiagonal = all(abs(A[abs(offset) > 1]) <= tol)
  # Orthogonal: the cross product is the identity
  orthogonal = max(abs(crossprod(A) - diag(nrow(A)))) <= tol
  # Positive definite: symmetric with positive eigenvalues
  posDef = symmetric && min(eigen(A, symmetric = TRUE, only.values = TRUE)$values) > tol
  c(square = TRUE, symmetric = symmetric, skew = skew, diagonal = diagonal, upper = upper,
    lower = lower, tridiagonal = tridiagonal, orthogonal = orthogonal, posDef = posDef)
}

#' @title Profile of a square matrix
#' @param A A square numeric matrix.
#' @param tol Tolerance used for the rank and for counting zero entries.
#' @return A list with elements `dim`, `rank`, `trace`, `det`, `sparsity`
#'   (the share of zero entries), `range`, `frobenius`, `asymmetry` (the
#'   Frobenius norm of the skew part relative to `frobenius`) and `nNA`.
#' @export
matrixProfile = function(A, tol = 1e-10) {
  if (!is.matrix(A) || !is.numeric(A)) {
    stop("`A` must be a numeric matrix.", call. = FALSE)
  }
  if (nrow(A) != ncol(A)) {
    stop("`A` must be square.", call. = FALSE)
  }
  nNA = sum(is.na(A))
  if (nNA > 0L) {
    return(list(dim = dim(A), rank = NA_integer_, trace = NA_real_, det = NA_real_,
                sparsity = NA_real_, range = c(NA_real_, NA_real_),
                frobenius = NA_real_, asymmetry = NA_real_, nNA = nNA))
  }

  frobenius = sqrt(sum(A^2))
  skewPart = (A - t(A)) / 2
  list(
    dim = dim(A),
    rank = qr(A, tol = tol)$rank,
    trace = sum(diag(A)),
    det = det(A),
    sparsity = mean(abs(A) <= tol),
    range = range(A),
    frobenius = frobenius,
    asymmetry = if (frobenius > 0) sqrt(sum(skewPart^2)) / frobenius else 0,
    nNA = nNA
  )
}

#' @title Summarise the spectrum of a symmetric matrix
#' @details Eigenvalues with absolute value at most `tol` count as zero. The
#'   condition number is `Inf` when any eigenvalue is zero.
#' @param A A symmetric numeric matrix.
#' @param tol Tolerance used to classify eigenvalues as zero.
#' @return A list with elements `values`, `radius`, `minAbs`, `nPositive`,
#'   `nNegative`, `nZero`, `inertia`, `trace`, `det` and `condition`.
#' @export
spectrumSummary = function(A, tol = 1e-10) {
  if (!isSymmetric(A)) {
    stop("`A` must be symmetric.", call. = FALSE)
  }
  values = eigen(A, symmetric = TRUE, only.values = TRUE)$values
  absValues = abs(values)

  # Size of the spectrum
  radius = max(absValues)
  minAbs = min(absValues)

  # Signs of the eigenvalues
  nPositive = sum(values > tol)
  nNegative = sum(values < -tol)
  nZero = sum(absValues <= tol)
  inertia = sprintf("(%d, %d, %d)", nPositive, nNegative, nZero)

  # Invariants recovered from the eigenvalues
  trace = sum(values)
  det = prod(values)
  condition = if (nZero > 0L) Inf else radius / minAbs

  list(values = values, radius = radius, minAbs = minAbs, nPositive = nPositive,
       nNegative = nNegative, nZero = nZero, inertia = inertia, trace = trace,
       det = det, condition = condition)
}

#' @title Tabulate statistics for a list of matrices
#' @param mats A named list of square numeric matrices.
#' @param stats Character vector naming the statistics; any of `"rank"`,
#'   `"trace"`, `"det"`, `"frobenius"`, `"maxAbs"` and `"condition"`.
#' @return A data frame with a `matrix` column and one column per statistic.
#' @export
summarizeMatrices = function(mats, stats = c("rank", "trace", "det", "frobenius", "maxAbs",
                                             "condition")) {
  if (is.null(names(mats))) {
    names(mats) = paste0("m", seq_along(mats))
  }
  out = data.frame(matrix = names(mats))
  for (stat in stats) {
    out[[stat]] = vapply(mats, function(A) {
      switch(stat,
        rank = {
          qr(A)$rank
        },
        trace = {
          sum(diag(A))
        },
        det = {
          det(A)
        },
        frobenius = {
          sqrt(sum(A^2))
        },
        maxAbs = {
          max(abs(A))
        },
        condition = {
          d = svd(A, nu = 0L, nv = 0L)$d
          if (min(d) == 0) Inf else max(d) / min(d)
        },
        stop(sprintf("Unknown statistic `%s`.", stat), call. = FALSE)
      )
    }, numeric(1), USE.NAMES = FALSE)
  }
  out
}
