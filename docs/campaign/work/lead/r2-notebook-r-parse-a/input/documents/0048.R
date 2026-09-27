acceptedMaxSSR <- function(
  CGNM_result,
  cutoff_pvalue = 0.05,
  numParametersIncluded = NA,
  useAcceptedApproximateMinimizers = TRUE,
  algorithm = 2
) {
  SSR_vec <- CGNM_result$residual_history[, dim(CGNM_result$residual_history)[
    2
  ]]

  if (useAcceptedApproximateMinimizers) {
    if (algorithm == 2) {
      numInInitialSet <- dim(CGNM_result$residual_history)[1]

      orderedIndex <- order(SSR_vec)

      bestIndex <- topIndices(CGNM_result, 1)

      ptest_vec <- c()

      cumResidualFromBestFit <- c()

      pvalue_vec <- c()
      for (i in seq(2, numInInitialSet)) {
        numInSample <- i - 1
        cumResidualFromBestFit <- c(
          cumResidualFromBestFit,
          sum(
            (CGNM_result$Y[orderedIndex[i - 1], ] -
              CGNM_result$Y[orderedIndex[i], ])
          )
        )

        if (numInSample > 3) {
          tStat <- qt(
            1 - cutoff_pvalue / (2 * numInSample),
            df = (numInSample - 2)
          )

          #test statistics from Grubbs' Test for Outliers
          testStatistics <- (numInSample - 1) /
            sqrt(numInSample) *
            sqrt(tStat^2 / (numInSample - 2 + tStat^2))

          #test hypothesis where the null hypothesis is that the last added sum of residual is outlier compared to the ones that are already accepted
          ptest_vec <- c(
            ptest_vec,
            abs(
              mean(cumResidualFromBestFit) -
                cumResidualFromBestFit[length(cumResidualFromBestFit)]
            ) /
              sd(cumResidualFromBestFit) >
              testStatistics
          )
        } else {
          ptest_vec <- c(ptest_vec, FALSE)
        }
      }

      acceptMaxSSR <- sort(SSR_vec)[which(ptest_vec)[1] + 1]
    } else {
      min_R <- min(SSR_vec)
      minIndex <- which(SSR_vec == min_R)[1]

      targetVector <- as.numeric(CGNM_result$runSetting$targetVector)
      targetVector[is.na(targetVector)] <- 0

      residual_vec <- CGNM_result$Y[minIndex, ] - targetVector

      acceptMaxSSR <- max(
        qchisq(1 - cutoff_pvalue, df = length(residual_vec)) *
          (sd(residual_vec))^2 +
          min_R,
        sqrt(sqrt(.Machine$double.eps))
      )

      accept_index <- which(SSR_vec < acceptMaxSSR)

      accept_vec <- as.vector(SSR_vec < acceptMaxSSR)

      numAccept <- sum(accept_vec, na.rm = TRUE)

      if (
        !is.na(numParametersIncluded) &
          (sum(accept_vec) > numParametersIncluded)
      ) {
        sortedAcceptedSSR <- sort(SSR_vec[accept_vec])[seq(
          1,
          numParametersIncluded
        )]
      } else {
        sortedAcceptedSSR <- sort(SSR_vec[accept_vec])
      }

      trapizoido_area <- c()
      for (i in seq(1, length(sortedAcceptedSSR) - 1)) {
        trapizoido_area <- c(
          trapizoido_area,
          (sortedAcceptedSSR[1] + sortedAcceptedSSR[i]) *
            i +
            (sortedAcceptedSSR[i + 1] +
              sortedAcceptedSSR[length(sortedAcceptedSSR)]) *
              (length(sortedAcceptedSSR) - i)
        )
      }

      strictMaxAcceptSSR <- sortedAcceptedSSR[which(
        trapizoido_area == min(trapizoido_area)
      )]

      accept_index <- which(SSR_vec < strictMaxAcceptSSR)
      accept_vec <- as.vector(SSR_vec < strictMaxAcceptSSR)
      numAccept <- sum(accept_vec, na.rm = TRUE)
      acceptMaxSSR <- strictMaxAcceptSSR
    }
    ## numPara*log(R_nu)+qchisq(1-alpha,1)
  } else if (
    is.na(numParametersIncluded) | numParametersIncluded > length(SSR_vec)
  ) {
    acceptMaxSSR <- max(SSR_vec, na.rm = TRUE)
  } else {
    acceptMaxSSR <- sort(SSR_vec)[numParametersIncluded]
  }

  acceptMaxSSR <- max(acceptMaxSSR, min(SSR_vec) + sqrt(.Machine$double.eps))
  return(acceptMaxSSR)
}