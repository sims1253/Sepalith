DyspiaSimpleImpl <- function(pathwayScores, pathwaysSizes, pathwaysFiltered,
                            leadingEdges, permPerProc, seeds,
                            toKeepLength, stats, BPPARAM){
    K <- max(pathwaysSizes)
    universe <- seq_along(stats)

    counts <- bplapply(seq_along(permPerProc), function(i) {
        nperm1 <- permPerProc[i]
        leEs <- rep(0, toKeepLength)
        geEs <- rep(0, toKeepLength)
        leZero <- rep(0, toKeepLength)
        geZero <- rep(0, toKeepLength)
        leZeroSum <- rep(0, toKeepLength)
        geZeroSum <- rep(0, toKeepLength)
        if (toKeepLength == 1) {
            for (i in seq_len(nperm1)) {
                randSample <- sample.int(length(universe), K)
                randEsP <- calcDyspiaStat(
                    stats = stats,
                    selectedStats = randSample,
                    DyspiaParam = 1)
                leEs <- leEs + (randEsP <= pathwayScores)
                geEs <- geEs + (randEsP >= pathwayScores)
                leZero <- leZero + (randEsP <= 0)
                geZero <- geZero + (randEsP >= 0)
                leZeroSum <- leZeroSum + pmin(randEsP, 0)
                geZeroSum <- geZeroSum + pmax(randEsP, 0)
            }
        } else {
            aux <- calcDyspiaStatCumulativeBatch(
                stats = stats,
                DyspiaParam = 1,
                pathwayScores = pathwayScores,
                pathwaysSizes = pathwaysSizes,
                iterations = nperm1,
                seed = seeds[i])
            leEs = get("leEs", aux)
            geEs = get("geEs", aux)
            leZero = get("leZero", aux)
            geZero = get("geZero", aux)
            leZeroSum = get("leZeroSum", aux)
            geZeroSum = get("geZeroSum", aux)
        }
        data.table(pathway=seq_len(toKeepLength),
                   leEs=leEs, geEs=geEs,
                   leZero=leZero, geZero=geZero,
                   leZeroSum=leZeroSum, geZeroSum=geZeroSum
                   )
    }, BPPARAM=BPPARAM)

    counts <- rbindlist(counts)

    # Getting rid of check NOTEs
    leEs=leZero=geEs=geZero=leZeroSum=geZeroSum=NULL
    pathway=padj=pval=DysPS=NDysPS=geZeroMean=leZeroMean=NULL
    nMoreExtreme=nGeEs=nLeEs=size=nLeZero=nGeZero=NULL
    leadingEdge=NULL
    .="damn notes"

    pvals <- counts[, list(leZeroMean = sum(leZeroSum) / sum(leZero),
                           geZeroMean = sum(geZeroSum) / sum(geZero),
                           nLeZero = sum(leZero),
                           nGeZero = sum(geZero),
                           nLeEs = sum(leEs),
                           nGeEs = sum(geEs)),
                    by = .(pathway)]

    pvals[, DysPS := pathwayScores[pathway]]

    pvals[, NDysPS := as.numeric(NA)]
    pvals[(DysPS > 0 & geZeroMean != 0) | (DysPS <= 0 & leZeroMean != 0),
          NDysPS := DysPS / ifelse(DysPS > 0, geZeroMean, abs(leZeroMean))]

    pvals[, pval := as.numeric(NA)]
    pvals[!is.na(NDysPS), pval := pmin((1+nLeEs) / (1 + nLeZero),
                        (1+nGeEs) / (1 + nGeZero))]

    pvals[, padj := as.numeric(NA)]
    pvals[!is.na(pval), padj := p.adjust(pval, method = "BH")]

    pvals[, nMoreExtreme :=  ifelse(DysPS > 0, nGeEs, nLeEs)]
    pvals[, size := pathwaysSizes[pathway]]
    pvals[, pathway := names(pathwaysFiltered)[pathway]]
    pvals[, leadingEdge := .(leadingEdges)]

    pvals
}