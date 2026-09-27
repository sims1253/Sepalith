panel2cs <- function(data, timevars, idname, tname) {
  if (length(unique(data[, tname])) != 2) {
    stop("panel2cs only for 2 periods of panel data")
  }

  # balance the data, just in case
  data <- make_balanced_panel(data, idname, tname)

  # put everything in the right order,
  # so we can match it easily later on
  data <- data[order(data[, idname], data[, tname]), ]

  tdta <- aggregate(
    data[, timevars],
    by = list(data[, idname]),
    FUN = function(x) {
      x[2]
    }
  )

  t1 <- unique(data[, tname])
  t1 <- t1[order(t1)][1]
  retdat <- subset(data, data[, tname] == t1)
  retdat$yt1 <- tdta[, 2]
  retdat$dy <- retdat$yt1 - retdat$y
  return(retdat)
}