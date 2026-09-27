.lec.init <- function() {

  if (exists(".lec.Random.seed.table", envir = .GlobalEnv)) {
    .lec.Random.seed.table
  } else {
    .lec.Random.seed.table <- sample.int(2^32, 2^32, replace = TRUE)
    .lec.Random.seed.table
  }
}