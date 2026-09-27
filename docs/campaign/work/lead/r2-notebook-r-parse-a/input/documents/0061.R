eigs <- function(
  M,
  nev = min(dim(M)[1] - 1, 1),
  sym = sum(abs(M - t(M))) / sum(abs(M)) < 1e-10,
  which = "LM",
  use.arpack = TRUE,
  options.arpack = NULL
) {
  if (all(class(M) != "matrix")) {
    stop("Input M must be a matrix.")
  }

  n <- dim(M)[1]

  if (nev > n) {
    stop("nev must be less than dimension of M")
  }

  if (use.arpack && requireNamespace("igraph", quietly = TRUE)) {
    options.arpack$which <- which
    options.arpack$n <- n
    options.arpack$nev <- nev
    options.arpack$ncv <- nev + 2

    #browser()
    ff <- function(x, extra) {
      extra %*% x
    }

    v <- igraph::arpack(ff, extra = M, sym = sym, options = options.arpack)
  } else {
    warning(
      "Using eigen. May be slow.  Use arpack in igraph package to improve speed."
    )
    v <- eigen(M, symmetric = sym)

    I <- switch(
      which,
      LM = 1:nev,
      SM = n:(n - nev + 1),
      stop("which can only be 'LM' or 'SM' when using eigen")
    )

    v$values <- v$values[I]
    v$vectors <- v$vectors[, I]
  }

  return(v)
}