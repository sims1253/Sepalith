AmatrixPolyCross <- function(data = NULL, fixedParent = FALSE) {
  unk <- 0
  orig.order <- as.character(data[, 1])

  data1 <- AGHmatrix::datatreat(data = data[, c(1, 2, 3)], unk = unk)
  Parents <- data1$sire
  Parents <- rbind(Parents, data1$dire)

  for (i in 4:ncol(data)) {
    Parents <- rbind(
      Parents,
      AGHmatrix::datatreat(data = data[, c(1, 2, i)], unk = unk)$dire
    )
  }

  for (i in seq_len(ncol(Parents))) {
    tst <- which(Parents[, i] != 0)
    if (length(tst) > 0) {
      if (min(which(Parents[, i] == 0)) < max(tst)) {
        stop(deparse(paste0(
          "Check parent order of line ",
          i,
          ", missing (or non-used) values should be located at the right!"
        )))
      }
    }
  }

  if (length(data$sire) > 1000) {
    cat(
      "Processing a large pedigree data... It may take a couple of minutes... \n"
    )
  }

  if (!fixedParent) {
    n <- ncol(Parents)
    Time <- proc.time()

    combs <- NULL
    for (i in 2:nrow(Parents)) {
      combs <- cbind(combs, combn(i, 2))
    }

    combs <- t(unique(t(combs))) #fixing order by number of max parents

    cat("Constructing matrix A using ploidy = 2 \n")
    Afinal <- matrix(NA, n, n)
    Afinal[1, 1] <- 1
    Acombs <- array(NA, dim = c(n, n, ncol(combs)))

    for (i in 2:n) {
      for (combsIndex in seq_len(ncol(combs))) {
        A <- Afinal
        s <- Parents[combs[1, combsIndex], ]
        d <- Parents[combs[2, combsIndex], ]
        if (s[i] == 0 && d[i] == 0) {
          A[i, i] <- 1
          for (j in 1:(i - 1)) {
            A[j, i] <- A[i, j] <- 0
          }
        }
        if (s[i] == 0 && d[i] != 0) {
          A[i, i] <- 1
          for (j in 1:(i - 1)) {
            A[j, i] <- A[i, j] <- 0.5 *
              (A[j, d[i]])
          }
        }
        if (d[i] == 0 && s[i] != 0) {
          A[i, i] <- 1
          for (j in 1:(i - 1)) {
            A[j, i] <- A[i, j] <- 0.5 *
              (A[j, s[i]])
          }
        }
        if (d[i] != 0 && s[i] != 0) {
          A[i, i] <- 1 + 0.5 * (A[d[i], s[i]])
          for (j in 1:(i - 1)) {
            A[j, i] <- A[i, j] <- 0.5 *
              (A[j, s[i]] + A[j, d[i]])
          }
        }
        Acombs[1:i, 1:i, combsIndex] <- A[1:i, 1:i]
      }

      ## Total parents
      totalParents <- length(which(Parents[, i] != 0))
      if (totalParents < 3) {
        Afinal[i, ] <- Afinal[, i] <- Acombs[i, , 1]
      } else {
        Afinal[i, ] <- Afinal[, i] <- rowMeans(Acombs[i, , (1:choose(totalParents, 2))])
      }
    }
    A <- Afinal
  }

  if (fixedParent) {
    n <- ncol(Parents)
    Time <- proc.time()
    Afinal <- matrix(NA, n, n)
    Afinal[1, 1] <- 1
    Acombs <- array(NA, dim = c(n, n, (nrow(Parents) - 1)))
    s <- Parents[1, ]
    for (i in 2:n) {
      for (combsIndex in 1:(nrow(Parents) - 1)) {
        A <- Afinal
        d <- Parents[combsIndex + 1, ]
        if (s[i] == 0 && d[i] == 0) {
          A[i, i] <- 1
          for (j in 1:(i - 1)) {
            A[j, i] <- A[i, j] <- 0
          }
        }
        if (s[i] == 0 && d[i] != 0) {
          A[i, i] <- 1
          for (j in 1:(i - 1)) {
            A[j, i] <- A[i, j] <- 0.5 *
              (A[j, d[i]])
          }
        }
        if (d[i] == 0 && s[i] != 0) {
          A[i, i] <- 1
          for (j in 1:(i - 1)) {
            A[j, i] <- A[i, j] <- 0.5 *
              (A[j, s[i]])
          }
        }
        if (d[i] != 0 && s[i] != 0) {
          A[i, i] <- 1 + 0.5 * (A[d[i], s[i]])
          for (j in 1:(i - 1)) {
            A[j, i] <- A[i, j] <- 0.5 *
              (A[j, s[i]] + A[j, d[i]])
          }
        }
        Acombs[1:i, 1:i, combsIndex] <- A[1:i, 1:i]
      }

      ## Total parents
      totalParents <- length(which(Parents[, i] != 0))
      if (totalParents < 3) {
        Afinal[i, ] <- Afinal[, i] <- Acombs[i, , 1]
      } else {
        Afinal[i, ] <- Afinal[, i] <- rowMeans(Acombs[i, , 1:(totalParents - 1)])
      }
    }
    A <- Afinal
  }

  Time <- as.matrix(proc.time() - Time)
  cat("Completed! Time =", Time[3] / 60, " minutes \n")
  rownames(A) <- colnames(A) <- data1$ind_data
  A <- A[orig.order, orig.order]
  return(A)
}