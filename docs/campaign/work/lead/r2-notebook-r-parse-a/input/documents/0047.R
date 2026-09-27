makeParaDistributionPlotDataFrame <- function(
  CGNM_result,
  indicesToInclude = NA,
  cutoff_pvalue = 0.05,
  numParametersIncluded = NA,
  ParameterNames = NA,
  ReparameterizationDef = NA,
  useAcceptedApproximateMinimizers = TRUE
) {
  ReReparameterise <- TRUE

  Re_ReparameterizationDef <- ReparameterizationDef
  Re_ParameterNames <- ParameterNames

  if (is.null(CGNM_result$runSetting$ReparameterizationDef)) {
    ReparameterizationDef <- paste0(
      "x",
      seq(1, length(CGNM_result$runSetting$initial_lowerRange))
    )
  } else {
    ReparameterizationDef <- CGNM_result$runSetting$ReparameterizationDef
  }

  if (
    length(CGNM_result$runSetting$ParameterNames) !=
      length(ReparameterizationDef)
  ) {
    ParameterNames <- ReparameterizationDef
  } else {
    ParameterNames <- CGNM_result$runSetting$ParameterNames
  }

  if (is.na(Re_ReparameterizationDef)[1]) {
    ReReparameterise <- FALSE
  }

  if (
    length(Re_ParameterNames) != length(Re_ReparameterizationDef) |
      is.na(Re_ParameterNames)[1]
  ) {
    Re_ParameterNames <- Re_ReparameterizationDef
  }

  Kind_iter <- NULL
  X_value <- NULL
  cluster <- NULL
  freeParaValues <- data.frame()
  SSR_vec <- CGNM_result$residual_history[, dim(CGNM_result$residual_history)[
    2
  ]]

  # if(is.na(ParameterNames[1])|length(ParameterNames)!=length(ReparameterizationDef)){
  #   ParameterNames=ReparameterizationDef
  # }
  #
  # if(is.na(ParameterNames[1])){
  #   ParameterNames=paste0("x",seq(1,dim(CGNM_result$initialX)[2]))
  #   ReparameterizationDef=ParameterNames
  # }

  if (!is.na(indicesToInclude[1])) {
    useIndecies_b <- seq(1, dim(CGNM_result$X)[1]) %in% indicesToInclude
  } else if (useAcceptedApproximateMinimizers) {
    useIndecies_b <- acceptedIndices_binary(
      CGNM_result,
      cutoff_pvalue,
      numParametersIncluded,
      useAcceptedApproximateMinimizers
    )
  } else {
    if (
      is.na(numParametersIncluded) |
        numParametersIncluded > dim(CGNM_result$X)[1]
    ) {
      useIndecies_b <- rep(TRUE, dim(CGNM_result$X)[1])
    } else {
      useIndecies_b <- (SSR_vec <= sort(SSR_vec)[numParametersIncluded])
    }
  }

  initialX_df <- CGNM_result$initialX
  colnames(initialX_df) <- paste0("x", seq(1, dim(CGNM_result$initialX)[2]))
  initialX_df <- data.frame(initialX_df)

  repara_initialX_df <- data.frame(row.names = seq(1, dim(initialX_df)[1]))
  temp_repara_initialX_df <- data.frame(row.names = seq(1, dim(initialX_df)[1]))

  for (i in seq(1, length(ParameterNames))) {
    temp_repara_initialX_df[, ParameterNames[i]] <- with(
      initialX_df,
      eval(parse(text = ReparameterizationDef[i]))
    )
  }
  if (ReReparameterise) {
    for (i in seq(1, length(Re_ParameterNames))) {
      repara_initialX_df[, Re_ParameterNames[i]] <- with(
        temp_repara_initialX_df,
        eval(parse(text = Re_ReparameterizationDef[i]))
      )
    }
  } else {
    repara_initialX_df <- temp_repara_initialX_df
  }

  finalX_df <- CGNM_result$X
  colnames(finalX_df) <- paste0("x", seq(1, dim(CGNM_result$X)[2]))
  finalX_df <- data.frame(finalX_df)

  repara_finalX_df <- data.frame(row.names = seq(1, dim(finalX_df)[1]))
  temp_repara_finalX_df <- data.frame(row.names = seq(1, dim(finalX_df)[1]))

  for (i in seq(1, length(ParameterNames))) {
    temp_repara_finalX_df[, ParameterNames[i]] <- with(
      finalX_df,
      eval(parse(text = ReparameterizationDef[i]))
    )
  }
  if (ReReparameterise) {
    for (i in seq(1, length(Re_ParameterNames))) {
      repara_finalX_df[, Re_ParameterNames[i]] <- with(
        temp_repara_finalX_df,
        eval(parse(text = Re_ReparameterizationDef[i]))
      )
    }
  } else {
    repara_finalX_df <- temp_repara_finalX_df
  }

  for (i in seq(1, dim(repara_initialX_df)[2])) {
    freeParaValues <- rbind(
      freeParaValues,
      data.frame(
        Name = names(repara_initialX_df)[i],
        X_value = repara_initialX_df[, i],
        Kind_iter = "Initial",
        SSR = NA
      )
    )
  }

  for (i in seq(1, dim(repara_finalX_df)[2])) {
    freeParaValues <- rbind(
      freeParaValues,
      data.frame(
        Name = names(repara_finalX_df)[i],
        X_value = repara_finalX_df[useIndecies_b, i],
        Kind_iter = "Final Accepted",
        SSR = SSR_vec[useIndecies_b]
      )
    )
  }

  if (!is.null(CGNM_result$bootstrapX)) {
    bootstrapX_df <- CGNM_result$bootstrapX
    colnames(bootstrapX_df) <- paste0(
      "x",
      seq(1, dim(CGNM_result$bootstrapX)[2])
    )
    bootstrapX_df <- data.frame(bootstrapX_df)

    repara_bootstrapX_df <- data.frame(
      row.names = seq(1, dim(bootstrapX_df)[1])
    )
    temp_repara_bootstrapX_df <- data.frame(
      row.names = seq(1, dim(bootstrapX_df)[1])
    )
    for (i in seq(1, length(ParameterNames))) {
      temp_repara_bootstrapX_df[, ParameterNames[i]] <- with(
        bootstrapX_df,
        eval(parse(text = ReparameterizationDef[i]))
      )
    }
    if (ReReparameterise) {
      for (i in seq(1, length(Re_ParameterNames))) {
        repara_bootstrapX_df[, Re_ParameterNames[i]] <- with(
          temp_repara_bootstrapX_df,
          eval(parse(text = Re_ReparameterizationDef[i]))
        )
      }
    } else {
      repara_bootstrapX_df <- temp_repara_bootstrapX_df
    }

    for (i in seq(1, dim(repara_bootstrapX_df)[2])) {
      freeParaValues <- rbind(
        freeParaValues,
        data.frame(
          Name = names(repara_bootstrapX_df)[i],
          X_value = repara_bootstrapX_df[, i],
          Kind_iter = "Bootstrap",
          SSR = NA
        )
      )
    }

    freeParaValues$Kind_iter <- factor(
      freeParaValues$Kind_iter,
      levels = c("Initial", "Final Accepted", "Bootstrap")
    )
  } else {
    freeParaValues$Kind_iter <- factor(
      freeParaValues$Kind_iter,
      levels = c("Initial", "Final Accepted")
    )
  }

  freeParaValues$Name <- factor(
    freeParaValues$Name,
    levels = names(repara_finalX_df)
  )

  return(freeParaValues)
}