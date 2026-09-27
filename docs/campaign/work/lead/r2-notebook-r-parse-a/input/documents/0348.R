createAnalysisDetails <- function(
  analysisId,
  sqlFileName,
  parameters,
  includedCovariateConceptIds = c(),
  addDescendantsToInclude = FALSE,
  excludedCovariateConceptIds = c(),
  addDescendantsToExclude = FALSE,
  includedCovariateIds = c()
) {
  analysisDetail <- list()
  formalNames <- names(formals(createAnalysisDetails))
  for (name in formalNames) {
    value <- get(name)
    if (is.null(value)) {
      value <- vector()
    }
    analysisDetail[[name]] <- value
  }
  class(analysisDetail) <- "analysisDetail"
  return(analysisDetail)
}