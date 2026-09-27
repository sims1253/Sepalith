getnextevnt <- function(simlist) {
  # find earliest event
  etimes <- simlist$evnts[, 1]
  whichnexte <- which.min(etimes)
  nextetime <- etimes[whichnexte]
  if (simlist$aevntset) {
    nextatime <- simlist$aevnts[simlist$nextaevnt, 1]
    if (nextatime < nextetime) {
      oldrow <- simlist$nextaevnt
      simlist$nextaevnt <- oldrow + 1
      return(simlist$aevnts[oldrow, ])
    }
  }
  # either don't have a separate arrivals event set, or the next
  # arrival is later than now
  head <- simlist$evnts[whichnexte, ]
  simlist$evnts[whichnexte, 1] <- simlist$timelim2
  return(head)
}