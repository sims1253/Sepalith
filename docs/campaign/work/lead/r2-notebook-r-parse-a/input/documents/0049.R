acceptedApproximateMinimizers <- function(
  CGNM_result,
  cutoff_pvalue = 0.05,
  numParametersIncluded = NA,
  useAcceptedApproximateMinimizers = TRUE,
  algorithm = 2,
  ParameterNames = NA,
  ReparameterizationDef = NA
) {
  out <- CGNM_result$X[
    acceptedIndices(
      CGNM_result,
      cutoff_pvalue,
      numParametersIncluded,
      useAcceptedApproximateMinimizers,
      algorithm = algorithm
    ),
  ]

  out <- data.frame(out)
  colnames(out) <- paste0("x", seq(1, dim(CGNM_result$initialX)[2]))

  if (
    is.na(ParameterNames)[1] & !is.null(CGNM_result$runSetting$ParameterNames)
  ) {
    ParameterNames <- CGNM_result$runSetting$ParameterNames
  }

  if (
    is.na(ReparameterizationDef)[1] &
      !is.null(CGNM_result$runSetting$ReparameterizationDef)
  ) {
    ReparameterizationDef <- CGNM_result$runSetting$ReparameterizationDef
  }

  if (
    is.na(ParameterNames[1]) |
      length(ParameterNames) != length(ReparameterizationDef)
  ) {
    ParameterNames <- ReparameterizationDef
  }

  if (is.na(ParameterNames[1])) {
    ParameterNames <- paste0("x", seq(1, dim(CGNM_result$initialX)[2]))
    ReparameterizationDef <- ParameterNames
  }

  repara_finalX_df <- data.frame(row.names = seq(1, dim(out)[1]))

  for (i in seq(1, length(ParameterNames))) {
    repara_finalX_df[, ParameterNames[i]] <- with(
      out,
      eval(parse(text = ReparameterizationDef[i]))
    )
  }

  return(repara_finalX_df)
}