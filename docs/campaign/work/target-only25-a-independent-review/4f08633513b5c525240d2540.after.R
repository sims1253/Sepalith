.lec.init <- function() {

  if (exists(".lec.Random.seed.table", envir = .GlobalEnv)) {
    return()
  }

  .lec.Random.seed.table <- .Random.seed.table
  .Random.seed.table <- NULL
}