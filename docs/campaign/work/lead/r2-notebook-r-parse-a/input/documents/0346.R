createDetailedCovariateSettings <- function(analyses = list()) {
  covariateSettings <- list(
    temporal = FALSE,
    temporalSequence = FALSE,
    analyses = analyses
  )
  attr(covariateSettings, "fun") <- "getDbDefaultCovariateData"
  class(covariateSettings) <- "covariateSettings"
  return(covariateSettings)
}