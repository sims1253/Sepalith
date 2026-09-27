simulated.sigma <- function(model.attributes, hyp.parm, parm, data) {
  X.matrix <- apply(parm$X, 2, c)
  data.matrix <- apply(data, 2, c)
  n.aux <- hyp.parm$n.sigma + model.attributes$S * model.attributes$times
  difference <- data.matrix - X.matrix %*% t(parm$B)
  n.s2.aux <- hyp.parm$n.s2.sigma + colSums(difference^2)
  parm$sigma2 <- 1 /
    rgamma(model.attributes$R, shape = n.aux / 2, rate = n.s2.aux / 2)

  return(parm)
}