.lec.init <- function() {

  if (exists(".lec.Random.seed.table", envir = .GlobalEnv)) {
    return(TRUE)
  }
  if (exists(".lec.Random.seed", envir = .GlobalEnv)) {
    return(FALSE)
  }
  return(TRUE)
}