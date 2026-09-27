uniAR <- function(data, scale = TRUE, winsize = 50, p = 1, dt = 1) {
  if (NCOL(data) != 2) {
    stop("Data requires only a time column and a timeseries")
  }
  if (!all(is.numeric(data[, -1]))) {
    stop("Not all time series are numeric")
  }
  data <- as.data.frame(data)

  window <- round(dim(data)[1] * winsize / 100)

  out <- lapply(1:(dim(data)[1] - window + 1), function(i) {
    sub_data <- data[i:(i + window - 1), ]
    if (isTRUE(scale)) {
      sub_data[, 2] <- c(scale(sub_data[, 2]))
    }
    #Ar_mod <- stats::ar.ols(as.matrix(sub_data[,-1]), aic = FALSE, order.max = p, dmean = FALSE,intercept = FALSE)$ar
    Ar_mod <- stats::ar(
      as.matrix(sub_data[, -1]),
      aic = FALSE,
      order.max = p,
      dmean = FALSE,
      intercept = FALSE,
      method = "yule-walker"
    )$ar

    jac <- rbind(as.numeric(Ar_mod), cbind(diag(p - 1), rep(0, p - 1)))

    yy <- eigen(jac)
    jac_eig <- (1 / dt) * (log(abs(yy$values)))
    return(cbind("time" = data[(i + window - 1), 1], "uniAR" = max(jac_eig)))
  })
  out <- as.data.frame(do.call("rbind", out))

  return(out)
}