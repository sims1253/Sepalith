DysPIA <- function(pathwayDB="kegg", stats, 
                   nperm=10000, minSize=15, maxSize=1000, 
                   nproc=0, DyspiaParam=1, BPPARAM=NULL) {

    pathwayDB <- tolower(pathwayDB)
    # Error if pathwayDB (pathway database name) is not correct
    {
        if (pathwayDB == "reactome")
            pathways <- pathway_list[[1]]
        else if (pathwayDB == "kegg")
            pathways <- pathway_list[[2]]
        else if (pathwayDB == "biocarta")
            pathways <- pathway_list[[3]]
        else if (pathwayDB == "panther")
            pathways <- pathway_list[[4]]
        else if (pathwayDB == "pathbank")
            pathways <- pathway_list[[5]]
        else if (pathwayDB == "nci")
            pathways <- pathway_list[[6]]
        else if (pathwayDB == "smpdb")
            pathways <- pathway_list[[7]]
        else if (pathwayDB == "pharmgkb")
            pathways <- pathway_list[[8]]
        else
            stop("The name of the pathway database you input is not correct!")
    }
    
    # Error if stats is not named
    if (is.null(names(stats))) {
        stop("stats should be named")
    }
    
    # Warning message for duplicate gene pair names
    if (any(duplicated(names(stats)))) {
        warning("There are duplicate gene pair names, DysPIA may produce unexpected results")
    }

    # Getting rid of check NOTEs
    leEs=leZero=geEs=geZero=leZeroSum=geZeroSum=NULL
    pathway=padj=pval=ES=NES=geZeroMean=leZeroMean=NULL
    nMoreExtreme=nGeEs=nLeEs=size=nLeZero=nGeZero=NULL
    leadingEdge=NULL
    
    
    granularity <- 1000
    permPerProc <- rep(granularity, floor(nperm / granularity))
    if (nperm - sum(permPerProc) > 0) {
        permPerProc <- c(permPerProc, nperm - sum(permPerProc))
    }
    seeds <- sample.int(10^9, length(permPerProc))

    BPPARAM <- setUpBPPARAM(nproc=nproc, BPPARAM=BPPARAM)

    minSize <- max(minSize, 1)
    stats <- sort(stats, decreasing=TRUE)

    stats <- abs(stats) ^ DyspiaParam
    pathwaysFiltered <- lapply(pathways, function(p) { as.vector(na.omit(fmatch(p, names(stats)))) })
    pathwaysSizes <- sapply(pathwaysFiltered, length)

    toKeep <- which(minSize <= pathwaysSizes & pathwaysSizes <= maxSize)
    m <- length(toKeep)

    if (m == 0) {
        return(data.table(pathway=character(),
                          pval=numeric(),
                          padj=numeric(),
                          DysPS=numeric(),
                          NDysPS=numeric(),
                          nMoreExtreme=numeric(),
                          size=integer(),
                          leadingEdge=list()))
    }

    pathwaysFiltered <- pathwaysFiltered[toKeep]
    pathwaysSizes <- pathwaysSizes[toKeep]

    DyspiaStatRes <- do.call(rbind,
                lapply(pathwaysFiltered, calcDyspiaStat,
                       stats=stats,
                       returnLeadingEdge=TRUE))

    leadingEdges <- mapply("[", list(names(stats)), DyspiaStatRes[, "leadingEdge"], SIMPLIFY = FALSE)
    pathwayScores <- unlist(DyspiaStatRes[, "res"])


    pvals <- DyspiaSimpleImpl(pathwayScores, pathwaysSizes, pathwaysFiltered,
                             leadingEdges, permPerProc, seeds, m, stats, BPPARAM)
    if (nrow(pvals[is.na(pval)]) > 0){
        warning("There were ",
                paste(nrow(pvals[is.na(pval)])),
                " pathways for which P-values were not calculated properly due to ",
                "unbalanced gene pair-level statistic values")
    }

    pvals[, nLeZero := NULL]
    pvals[, nGeZero := NULL]
    pvals[, leZeroMean := NULL]
    pvals[, geZeroMean := NULL]
    pvals[, nLeEs := NULL]
    pvals[, nGeEs := NULL]

    setcolorder(pvals, c("pathway", "pval", "padj", "DysPS", "NDysPS",
                         "nMoreExtreme", "size", "leadingEdge"))
    # Makes pvals object printable immediatly
    pvals <- pvals[]
    pvals
}