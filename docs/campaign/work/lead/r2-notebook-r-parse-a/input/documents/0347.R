createDetailedTemporalCovariateSettings <- function(
  analyses = list(),
  temporalStartDays = -365:-1,
  temporalEndDays = -365:-1
) {
  covariateSettings <- list(
    temporal = TRUE,
    temporalSequence = FALSE
  )
  formalNames <- names(formals(createDetailedTemporalCovariateSettings))
  for (name in formalNames) {
    covariateSettings[[name]] <- get(name)
  }
  attr(covariateSettings, "fun") <- "getDbDefaultCovariateData"
  class(covariateSettings) <- "covariateSettings"
  return(covariateSettings)
}