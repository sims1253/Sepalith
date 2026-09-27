class.L1 <- function(model, data) {
  aux <- mapply(
    function(m, d) {
      prd <- stats::predict(m, d, decision.values = TRUE)
      err <- rownames(d[prd != d$class, ])
      dst <- attr(prd, "decision.values")[err, ]
      sum(abs(dst)) / nrow(d)
    },
    m = model,
    d = data
  )

  aux <- 1 - 1 / (aux + 1)
  return(aux)
}