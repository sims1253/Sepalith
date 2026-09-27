buildSeed <- function(
  trainSet,
  genesInSeed = 200,
  groupSize = 30,
  randomize = TRUE,
  num.trees = 1000,
  plotIt = TRUE,
  trainSet.3sam = NULL,
  trainSet.30sam = NULL,
  proportional = FALSE
) {
  if (is.null(trainSet.3sam)) {
    trainSet.3sam <- ADAPTS::scSample(
      RNAcounts = trainSet,
      groupSize = 3,
      randomize = randomize
    )
  }
  if (proportional) {
    #colnames(trainSet) <- sub('\\.[0-9]+$', '', colnames(trainSet))
    tsNames <- sub('\\.[0-9]+$', '', colnames(trainSet))
    cellProps <- table(tsNames)
    cellSampleCounts <- 3 * ceiling(100 * cellProps / sum(cellProps))
    trainList <- list()
    for (curCount in unique(cellSampleCounts)) {
      curClusts <- names(cellSampleCounts)[cellSampleCounts == curCount]
      trainList[[as.character(curCount)]] <- ADAPTS::scSample(
        RNAcounts = trainSet[, tsNames %in% curClusts],
        groupSize = curCount,
        randomize = randomize
      )
    }
    trainSet.30sam <- do.call(cbind, trainList)
  } else {
    if (is.null(trainSet.30sam)) {
      trainSet.30sam <- ADAPTS::scSample(
        RNAcounts = trainSet,
        groupSize = groupSize,
        randomize = randomize
      )
    }
  }

  clusterIDs <- factor(colnames(trainSet.30sam))
  trainSet.4reg <- t(trainSet.30sam)

  rf1 <- ranger::ranger(
    x = trainSet.4reg,
    y = clusterIDs,
    num.trees = num.trees,
    importance = 'impurity'
  )
  imp <- ranger::importance(rf1)
  imp <- sort(imp[imp > 0], decreasing = TRUE)

  topGenes <- names(imp)[1:min(genesInSeed, length(imp))]
  #topGenes[!topGenes %in% rownames(trainSet.3sam)]

  seedMat <- trainSet.3sam[rownames(trainSet.3sam) %in% topGenes, ]
  cellTypes <- sub('\\.[0-9]+$', '', colnames(seedMat))
  seedMat <- t(apply(seedMat, 1, function(x) {
    tapply(x, cellTypes, mean, na.rm = TRUE)
  }))
  if (plotIt) {
    pheatmap::pheatmap(
      seedMat,
      main = paste(
        'Seed Matrix',
        '\n# Cell Types:',
        ncol(seedMat),
        '| # Genes:',
        nrow(seedMat)
      )
    )
  }
  return(seedMat)
}