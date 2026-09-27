CompareNumericSets <- function(
  DF,
  IndexA,
  numVars,
  IndexB = NULL,
  cLevel = 0.95
) {
  #
  stopifnot("DF must be a data frame" = is.data.frame(DF))
  #
  if (is.null(IndexB)) {
    IndexB <- setdiff(seq(1, nrow(DF), 1), IndexA)
  }
  outFrame <- NULL
  n <- length(numVars)
  stopifnot("numVars list is empty" = n > 0)
  for (i in 1:n) {
    upFrame <- WelchRankTest(DF, numVars[i], IndexA, IndexB, cLevel)
    outFrame <- rbind.data.frame(outFrame, upFrame)
  }
  names(outFrame) <- names(upFrame)
  outFrame$Variable <- numVars
  #
  allVars <- colnames(outFrame)
  varIndex <- which(allVars == "Variable")
  newIndex <- c(varIndex, setdiff(seq_len(ncol(outFrame)), varIndex))
  outFrame <- outFrame[, newIndex]
  #
  return(outFrame)
}