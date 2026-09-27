class.L3 <- function(model, data) {
  aux <- mapply(
    function(m, d) {
      tmp <- class.generate(d, nrow(d))
      prd <- stats::predict(m, tmp)
      error(prd, tmp$class)
    },
    m = model,
    d = data
  )

  return(aux)
}