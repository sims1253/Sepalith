read.DeponsBlockdyn <- function(
  fname,
  title = "NA",
  landscape = "NA",
  simtime = "NA",
  timestep = 30,
  startday = "2010-01-01",
  tz = "UTC"
) {
  # Get sim date and time from file name
  if (simtime == "NA") {
    simtime <- get.simtime(fname)
  }
  if (startday == "NA") {
    startday <- NA
  }
  all.data <- new("DeponsBlockdyn")
  all.data@title <- title
  all.data@landscape <- landscape
  all.data@simtime <- as.POSIXlt(simtime)
  all.data@startday <- as.POSIXlt(startday)
  the.data <- utils::read.csv(fname, sep = ",")
  names(the.data) <- c("tick", "block", "count")
  unique.blocks <- unique(the.data$block[1:min(nrow(the.data), 1000)]) # find number of different blocks (checking at most the first 1000 entries)
  real.time.1stblock <- tick.to.time(
    the.data$tick[the.data$block == unique.blocks[1]],
    timestep = timestep,
    origin = startday
  ) # calculate date for 1st block of each tick only
  the.data$real.time <- rep(real.time.1stblock, each = length(unique.blocks)) # replicate date for each other block of each tick; i.e., make n reps of each date, were n is number of blocks
  all.data@dyn <- the.data
  return(all.data)
}