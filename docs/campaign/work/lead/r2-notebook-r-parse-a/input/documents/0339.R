calc_cfi_from_vectors <- function(x, y, conf.level = 0.95, max_iter = NULL) {
  alpha <- 1 - conf.level
  p <- fast_welch_p(x, y)
  if (is.na(p) || p >= alpha) {
    return(0L)
  }

  if (is.null(max_iter)) {
    max_iter <- length(x) + length(y)
  }
  iter <- 0L

  repeat {
    if (mean(x) >= mean(y)) {
      high <- x
      low <- y
      from_x <- TRUE
    } else {
      high <- y
      low <- x
      from_x <- FALSE
    }
    m <- mean(high)
    above <- high[high > m]
    if (length(above) == 0L) {
      break
    }
    idx_in_above <- which.min(above - m)
    val <- above[idx_in_above]
    idx_in_high <- which(high == val)[1]

    high <- high[-idx_in_high]
    low <- c(low, val)

    if (from_x) {
      x <- high
      y <- low
    } else {
      y <- high
      x <- low
    }

    iter <- iter + 1L
    p <- fast_welch_p(x, y)
    if (is.na(p) || p >= alpha) {
      return(iter)
    }
    if (iter >= max_iter) return(NA_integer_)
  }
  iter
}