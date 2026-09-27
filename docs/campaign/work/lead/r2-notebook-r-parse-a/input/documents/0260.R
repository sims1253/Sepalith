AdjustPvalue <-
  function(tVal, tVec, pVec) {
    # True positives are considered to be anything above the threshold in the test population
    nTP <- length(which(tVec >= tVal))
    # The value is scaled to deal with different size test and permuted populations
    tpScale <- length(pVec) / length(tVec)
    nTP <- nTP * tpScale
    # False positives are considered to be anything above the threshold in the permuted population
    nFP <- length(which(pVec >= tVal))
    # FDR is caluclated using (FP)/(TP)
    adjP <- nFP / nTP
    return(adjP)
  }