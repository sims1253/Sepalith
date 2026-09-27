imputeConstant <- function(inputData, LOQ, constantValue) {
  if (missing(constantValue)) {
    constantValue <- LOQ / 2
  }
  imputedData <- inputData
  imputedData[inputData < LOQ] <- constantValue
  return(imputedData)
}