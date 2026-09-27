class.L2 <- function(model, data) {
  aux <- mapply(
    function(m, d) {
      prd <- stats::predict(m, d)
      error(prd, d$class)
    },
    m = model,
    d = data
  )

  return(aux)
}