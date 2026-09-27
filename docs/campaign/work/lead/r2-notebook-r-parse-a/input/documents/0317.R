getfreerow <- function(simlist) {
  evtimes <- simlist$evnts[, 1]
  tmp <- Position(simlist$passedtime, evtimes)
  if (is.na(tmp)) {
    stop('no room for new event')
  }
  tmp
}