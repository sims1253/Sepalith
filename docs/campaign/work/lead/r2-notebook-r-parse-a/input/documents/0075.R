cscov.shrbe <- function(
  comb = "shrbe",
  ...,
  n = NULL,
  p = NULL,
  matNA = NULL,
  res = NULL,
  mse = TRUE,
  shrink_fun = NULL
) {
  if (is.null(n)) {
    if (is.list(res)) {
      n <- NCOL(res[[1]])
    } else {
      cli_abort("Argument {.arg n} is NULL.", call = NULL)
    }
  }

  if (is.null(p)) {
    if (is.list(res)) {
      p <- length(res)
    } else {
      cli_abort("Argument {.arg p} is NULL.", call = NULL)
    }
  }

  if (is.null(res)) {
    cli_abort("Argument {.arg res} is NULL.", call = NULL)
  }

  if (is.list(res)) {
    res <- do.call(cbind, res)
  }
  id <- rep(1:p, each = n)
  if (!is.null(matNA)) {
    if (is.logical(matNA)) {
      ina <- as.vector(!matNA)
    } else {
      ina <- as.vector(is.na(matNA) | matNA != 0)
    }
    if (NCOL(res) != sum(ina)) {
      res <- res[, ina, drop = FALSE]
    }
    id <- id[ina]
  }

  if (is.null(shrink_fun)) {
    if (!is.null(matNA)) {
      if (sum(ina) == n * p) {
        shrink_fun <- shrink_estim
      } else {
        shrink_fun <- shrink_estim_na
      }
    } else {
      shrink_fun <- shrink_estim_na
    }
  }
  Wlist <- lapply(sort(unique(id)), function(i) {
    shrink_fun(res[, id == i, drop = FALSE], mse = mse)
  })
  lambda <- sapply(Wlist, attr, "lambda")
  Wlist <- bdiag(Wlist)
  attr(Wlist, "lambda") <- lambda
  return(Wlist)
}