NetworkCommunity <- function (networkeco, run = 100)
{
  returnlist <- list()
  returnlist[["RunNumber"]] <- run
  if (class(networkeco) == "qgraph" || class(networkeco) == "igraph") {
    Network <- networkeco
  } else { if (any(lapply(networkeco, function (x) class(x)) == "qgraph") ||
               any(lapply(networkeco, function (x) class(x)) == "igraph")) {
    if (any(lapply(networkeco, function (x) class(x)) == "qgraph") == TRUE) {
      Network <- networkeco[[as.numeric(which(lapply(networkeco, function (x) class(x)) == "qgraph"))]]
    } else {
      Network <- networkeco[[as.numeric(which(lapply(networkeco, function (x) class(x)) == "igraph"))]]
    }
  }  else {
    stop("networkeco must be (or contain) a qgraph or igraph object")
  }
  }
  if (class(Network) != "igraph") {
    Network <- igraph::as.igraph(Network, attributes = TRUE)
  }
  if (is.null(networkeco$Count) && is.null(networkeco$Distance) && is.null(networkeco$PercentDist)) {
    Node <- as.data.frame(Network[1])
  } else { if (is.null(networkeco$Count) && is.null(networkeco$Distance)) {
    Node <- networkeco$PercentDist
  } else { if (is.null(networkeco$Count)) {
    Node <- networkeco$Distance
  } else { Node <- networkeco$Count }
  }
  }
  SpinglassCommunities <- data.frame(matrix(NA, nrow = run, ncol = nrow(Node)+1,
                                         dimnames = list(1:run, c(rownames(Node), "MaxNumber"))))
  for (i in 1:run) {
    withr::with_seed(i,
    spinglassTest <- igraph::spinglass.community(Network)
    )
    SpinglassCommunities[i,1:(ncol(SpinglassCommunities)-1)] <- spinglassTest$membership
    SpinglassCommunities[i,ncol(SpinglassCommunities)] <- max(spinglassTest$membership)
  }
  for (i in 1:nrow(SpinglassCommunities)) {
    comvec <- as.numeric(SpinglassCommunities[i,-ncol(SpinglassCommunities)])
    comvec <- as.factor(comvec)
    ord <-  as.list(unique(comvec))
    names(ord) <- as.character(1:length(ord))
    levels(comvec) <- ord
    SpinglassCommunities[i,-ncol(SpinglassCommunities)] <- as.numeric(comvec)
  }
  returnlist[["Spinglass"]] <-  SpinglassCommunities
  FreqCommunity <- list()
  FreqCommunity[["ID"]] <- c(min(SpinglassCommunities$MaxNumber):max(SpinglassCommunities$MaxNumber))
  for (i in min(SpinglassCommunities$MaxNumber):max(SpinglassCommunities$MaxNumber)) {
    FreqCommunity[["Count"]][i] <- nrow(SpinglassCommunities[SpinglassCommunities$MaxNumber == i,])
    FreqCommunity[["Proportions"]][i] <- nrow(SpinglassCommunities[SpinglassCommunities$MaxNumber == i,])/nrow(SpinglassCommunities)
  }
  FreqCommunity[["Count"]] <- stats::na.omit(FreqCommunity[["Count"]])
  attributes(FreqCommunity[["Count"]])$na.action <- NULL
  FreqCommunity[["Proportions"]] <- stats::na.omit(FreqCommunity[["Proportions"]])
  attributes(FreqCommunity[["Proportions"]])$na.action <- NULL
  returnlist[["MaxCommunityNumber"]] <-  FreqCommunity
  Mean <- colMeans(as.matrix(SpinglassCommunities[,-ncol(SpinglassCommunities)]))
  RoundedMean <- round(Mean)
  returnlist[["Memberships"]][["Mean"]] <- Mean
  returnlist[["Memberships"]][["RoundedMean"]] <- RoundedMean
  return(returnlist)
}