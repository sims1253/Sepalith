residuals.cyclopsFit <- function(
  object,
  parm = NULL,
  type = "schoenfeld",
  ...
) {
  .checkInterface(object$cyclopsData, testOnly = TRUE)

  if (object$cyclopsData$modelType != "cox") {
    stop("Residuals for only Cox models are currently implemented")
  }
  if (type != "schoenfeld") {
    stop("Only Schoenfeld residuals are currently implemented")
  }

  if (getNumberOfCovariates(object$cyclopsData) != 1) {
    stop("Only single-covariate models are currently implemented")
  }

  res <- .cyclopsGetSchoenfeldResiduals(object$interface, NULL)

  res <- res[order(res$strata, res$times), ]

  result <- res$residuals
  names(result) <- res$times

  tbl <- table(res$strata)
  if (dim(tbl) > 1) {
    names(tbl) <- paste0("stratum=", names(tbl))
    attr(result, "strata") <- tbl
  }

  return(result)
}