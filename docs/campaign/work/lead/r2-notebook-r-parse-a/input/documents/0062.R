analyze_survival <- function(RD,
                             name = "no name",
                             xtreat = NULL,
                             C = 20) {
  if (!(class(RD)[1] %in% c("data.frame","matrix"))){
    stop("error: RD must be of class data.frame or matrix")
  }
  result <- list("name" = name)
  if (is.null(xtreat)) {
    result$"xtreat" <- 0:(ncol(RD) - 2)
  } else {
    if (length(xtreat) != (ncol(RD) - 1)) {
      stop("error: length of assigned treatments does not match data ")
    }
    result$"xtreat" <- xtreat
  }
  result$"raw" <- as.data.frame(RD)
  result$"mean" <-
    aggregate(
      x = result$raw,
      by = list(result$raw[, 1]),
      FUN = "mean",
      na.rm = TRUE
    )
  # store fit-summaries
  result$"fit" <- vector("list", dim(RD)[2] - 1)
  # store survival fractions
  result$"SF" <- rep(NA, dim(RD)[2] - 2)
  # summary of uncertainty analysis
  udf <-
    data.frame(
      "treatment" = result$"xtreat",
      "C" = C,
      "SF" = NA,
      "sd.SF" = NA,
      "lb.SF" = NA,
      "ub.SF" = NA,
      "log10.SF" = NA,
      "sd.log10.SF" = NA,
      "lb.log10.SF" = NA,
      "ub.log10.SF" = NA,
      "b" = NA,
      "sd.b" = NA
    )
  P0 <- pwr_reg(seeded = result$"mean"[, 2],
                counted = result$"mean"[, 3])
  result$"fit"[[1]] <- P0
  udf$"SF"[1] <- 1
  udf$"log10.SF"[1] <- 0
  udf$"b"[1] <- P0$coefficients[2,1]
  udf$"sd.b"[1] <- sqrt(vcov(P0)[2,2])
  # calculate SF for treatments with colony numbers overlapping with reference
  for (i in seq_along(result$"fit")[-1]) {
    Px <- pwr_reg(seeded = result$"mean"[, 2],
                  counted = result$"mean"[, i + 2])
    result$"fit"[[i]] <- Px
    ran_ref <- range(result$"raw"[, 2],na.rm = TRUE)
    ran_tre <- range(result$"raw"[, i + 1],na.rm = TRUE)
    if ( (ran_ref[2] >= ran_tre[1] ) & (ran_ref[1]) <= ran_tre[2] ){
      udf$"b"[i] <- Px$coefficients[2,1]
      udf$"sd.b"[i] <- sqrt(vcov(Px)[2,2])
      # calculate survival fraction
      sf <- calculate_sf(
        par_ref = P0,
        par_treat = Px,
        C = C
      )
      udf$"SF"[i] <- sf
      if(sf>0) {
        udf$"log10.SF"[i] <- log10(sf)
      }
      result$"SF"[i - 1] <- sf
      # calculate uncertainty (First-Order-Taylor-Series-Approximation)
      b0 <- P0$coefficients[2,1]
      z0 <- c(1, (log(C)-P0$coefficients[1,1])/b0)
      S0 <- vcov(P0)
      bx <- Px$coefficients[2,1]
      zx <- c(1, (log(C)-Px$coefficients[1,1])/bx)
      Sx <- vcov(Px)
      var_lsf <- (1/b0)^2*(z0%*%S0%*%z0) + (1/bx)^2*(zx%*%Sx%*%zx)
      # sd(log10(SF))
      udf$"sd.log10.SF"[i] <-  sqrt(var_lsf)/log(10)
      # sd(SF)
      udf$"sd.SF"[i] <- sf * sqrt(var_lsf)
    } else {
      warning("warning: SF calculation omitted, range of colonies counted
              in reference and treated cells do not overlap.")
      result$"SF"[[i - 1]] <- NaN
    }
  }
  udf$"lb.log10.SF" <- udf$"log10.SF" - 1.96 * udf$"sd.log10.SF"
  udf$"ub.log10.SF" <- udf$"log10.SF" + 1.96 * udf$"sd.log10.SF"
  udf$"lb.SF" <- 10^(udf$"lb.log10.SF")
  udf$"ub.SF" <- 10^(udf$"ub.log10.SF")

  result$"uncertainty" <- udf
  return(result)
}