.lec.init <- function() {

  if (exists(".lec.Random.seed.table", envir = .GlobalEnv)) {
    .lec.Random.seed.table
  } else {
    .lec.Random.seed.table <- sample.int(1000, 1000)
  }

  .lec.Random.seed.table
}