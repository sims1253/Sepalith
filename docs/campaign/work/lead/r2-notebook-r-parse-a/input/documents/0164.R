detc <- function(
  response = NULL,
  predictors = NULL,
  dets = NULL,
  names = NULL,
  conf = NULL,
  positive = "",
  parallel = FALSE,
  ncores = detectCores(),
  nboot = NULL,
  plot = FALSE,
  ...
) {
  if (is.null(dets)) {
    response <- as.factor(response)
  }

  if (!is.matrix(predictors)) {
    if (is.vector(predictors)) {
      ## if it's a vector, just change it to matrix with one column
      predictors <- matrix(predictors, ncol = 1)
    }
  }

  assertDetCurveParameters(
    response,
    predictors,
    ncores,
    conf,
    dets,
    names,
    nboot
  )
  if (!is.null(dets)) {
    if (isDetCurveAlreadyComputedForCI(dets, conf)) {
      if (plot) {
        plot.DETs(dets, ...)
      }
      return(dets)
    } else {
      response <- dets@detCurves[[1]]@response
    }
  }
  predictorList <- buildPredictorList(dets, predictors, names)
  nCurves <- length(predictorList)
  detCurvesInformation <- list()
  levels <- extractLevelsFromResponse(response, positive)
  defaultW <- getOption("warn")
  options(warn = -1)
  for (i in seq(nCurves)) {
    cat("Calculating DET Curve for:", names(predictorList)[i], "\n")
    if (parallel) {
      cluster <- makeCluster(ncores)
      registerDoParallel(cluster)
    }
    rocCurve <- roc(response, predictorList[[i]], levels = levels)
    if (!is.null(conf)) {
      if (is.null(nboot)) {
        nboot <- 2000
      }
      sensitivityConfidenceInterval <- ci.se(
        rocCurve,
        specificities = rocCurve$specificities,
        conf.level = conf,
        boot.n = as.integer(nboot),
        method = "bootstrap",
        parallel = parallel
      )
      detCurveInformation <- buildDetCurveInformationWithCI(
        rocCurve,
        conf,
        sensitivityConfidenceInterval
      )
    } else {
      detCurveInformation <- buildDetCurveInformationWithoutCI(rocCurve)
    }
    detCurvesInformation[[
      length(detCurvesInformation) + 1
    ]] <- detCurveInformation
    if (parallel) {
      stopCluster(cluster)
    }
  }
  options(warn = defaultW)
  names(detCurvesInformation) <- names(predictorList)
  detCurves <- (new("DETs", detCurves = detCurvesInformation))
  if (plot) {
    plot.DETs(detCurves, ...)
  }
  return(detCurves)
}