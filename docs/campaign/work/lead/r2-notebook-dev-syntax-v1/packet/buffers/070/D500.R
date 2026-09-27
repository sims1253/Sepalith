.lec.init <- function() {

  if (exists(".lec.Random.seed.table", envir = .GlobalEnv)) {
    .lec.Random.seed.table <- NULL
  }
  .lec.Random.seed.table <- sample.int(0:1, 1)
}