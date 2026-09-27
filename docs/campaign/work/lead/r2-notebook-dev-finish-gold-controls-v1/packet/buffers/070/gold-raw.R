.lec.init <- function() {

  if (exists(".lec.Random.seed.table", envir = .GlobalEnv)) {
    rm(".lec.Random.seed.table", envir = .GlobalEnv)
  }
  pos <- 1
  assign(
    ".lec.Random.seed.table",
    list(
      Cg = matrix(0, nrow = 0, ncol = 6),
      Bg = matrix(0, nrow = 0, ncol = 6),
      Ig = matrix(0, nrow = 0, ncol = 6),
      AIP = matrix(0, nrow = 0, ncol = 2),
      name = c()
    ),
    envir = as.environment(pos)
  )
  .Call("r_create_current_stream", PACKAGE = "rlecuyer")
  return(1)
}