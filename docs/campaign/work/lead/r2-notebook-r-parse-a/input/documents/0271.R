calcDyspiaStat <- function(stats, selectedStats, DyspiaParam=1,
                           returnAllExtremes=FALSE,
                           returnLeadingEdge=FALSE) {
    
    S <- selectedStats
    r <- stats
    p <- DyspiaParam
    
    S <- sort(S)
    
    m <- length(S)
    N <- length(r)
    if (m == N) {
        stop("DysPS statistic is not defined when all gene pairs are selected.")
    }
    NR <- (sum(abs(r[S])^p))
    rAdj <- abs(r[S])^p
    if (NR == 0) {
        # this is equivalent to rAdj being rep(eps, m)
        rCumSum <- seq_along(rAdj) / length(rAdj)
    } else {
        rCumSum <- cumsum(rAdj) / NR
    }
    
    
    tops <- rCumSum - (S - seq_along(S)) / (N - m)
    if (NR == 0) {
        # this is equivalent to rAdj being rep(eps, m)
        bottoms <- tops - 1 / m
    } else {
        bottoms <- tops - rAdj / NR
    }
    
    maxP <- max(tops)
    minP <- min(bottoms)
    
    if(maxP > -minP) {
        genePairSetStatistic <- maxP
    } else if (maxP < -minP) {
        genePairSetStatistic <- minP
    } else {
        genePairSetStatistic <- 0
    }
    
    if (!returnAllExtremes && !returnLeadingEdge) {
        return(genePairSetStatistic)
    }
    
    res <- list(res=genePairSetStatistic)
    if (returnAllExtremes) {
        res <- c(res, list(tops=tops, bottoms=bottoms))
    }
    if (returnLeadingEdge) {
        leadingEdge <- if (maxP > -minP) {
            S[seq_along(S) <= which.max(bottoms)]
        } else if (maxP < -minP) {
            rev(S[seq_along(S) >= which.min(bottoms)])
        } else {
            NULL
        }
        
        res <- c(res, list(leadingEdge=leadingEdge))
    }
    res
}