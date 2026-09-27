maxent.jar.args <- function(occs.z, bg.z, tune.tbl.i, other.settings) {
  out <- list()
  out$x <- rbind(occs.z, bg.z)
  out$p <- c(rep(1, nrow(occs.z)), rep(0, nrow(bg.z)))
  out$args <- c(
    "noremoveDuplicates",
    "noautofeature",
    other.settings$other.args
  )
  if (!grepl("L", tune.tbl.i$fc)) {
    out$args <- c(out$args, "nolinear")
  }
  if (!grepl("Q", tune.tbl.i$fc)) {
    out$args <- c(out$args, "noquadratic")
  }
  if (!grepl("H", tune.tbl.i$fc)) {
    out$args <- c(out$args, "nohinge")
  }
  if (!grepl("P", tune.tbl.i$fc)) {
    out$args <- c(out$args, "noproduct")
  }
  if (!grepl("T", tune.tbl.i$fc)) {
    out$args <- c(out$args, "nothreshold")
  } else {
    out$args <- c(out$args, "threshold")
  }
  out$args <- c(out$args, paste0("betamultiplier=", tune.tbl.i$rm, sep = ""))
  out$path <- other.settings$path
  return(out)
}