newsim <- function(
  timelim,
  maxesize,
  appcols = NULL,
  aevntset = FALSE,
  dbg = FALSE
) {
  simlist <- new.env()
  simlist$currtime <- 0.0 # current simulated time
  simlist$timelim <- timelim
  simlist$timelim2 <- 2 * timelim
  simlist$passedtime <- function(z) z > simlist$timelim
  simlist$evnts <-
    matrix(nrow = maxesize, ncol = 2 + length(appcols)) # event set
  colnames(simlist$evnts) <- c('evnttime', 'evnttype', appcols)
  simlist$evnts[, 1] <- simlist$timelim2
  simlist$aevntset <- aevntset
  if (aevntset) {
    simlist$aevnts <- NULL # will be reset by exparrivals()
    simlist$nextaevnt <- 1 # row number in aevnts of next arrival
  }
  simlist$dbg <- dbg
  simlist
}