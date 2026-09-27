qselection <- function(
  x,
  y,
  qvector,
  criterion = "deviance",
  method = "lm",
  family = "gaussian",
  nfolds = 5,
  cluster = TRUE,
  ncores = NULL
) {
  if (missing(x)) {
    stop("Argument \"x\" is missing, with no default")
  }
  if (missing(y)) {
    stop("Argument \"y\" is missing, with no default")
  }
  if (missing(qvector)) {
    stop("Argument \"qvector\" is missing, with no default")
  }

  in_c <- c()
  var <- c()
  qq <- c()
  cont <- 0
  res <- c()
  for (q in qvector) {
    cont <- cont + 1
    if (q == qvector[1]) {
      prevar <- NULL
    } else {
      prevar <- aux$Variable_numbers
    }
    aux <- selection(
      x = x,
      y = y,
      q = q,
      prevar = prevar,
      criterion = criterion,
      method = method,
      family = family,
      seconds = F,
      cluster = cluster,
      ncores = ncores,
      nfolds = nfolds
    )
    in_c[cont] <- round(aux$Information_Criterion, 3)
    var[cont] <- toString(aux$Variable_names)
    qq[cont] <- q
    print(paste("Selecting subset of size", q, "...", sep = " "))
  }
  res <- data.frame(qq, in_c, var)
  colnames(res) <- c("q", paste(criterion), "selection")
  class(res) <- "qselection"
  return(res)
}