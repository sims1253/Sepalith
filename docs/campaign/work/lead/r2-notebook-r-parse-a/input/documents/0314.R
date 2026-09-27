force.heredity.bas <- function(object, prior.prob = 0.5) {
  parents <- make.parents.of.interactions(
    mf = eval(object$call$formula, parent.frame()),
    data = eval(object$call$data, parent.frame())
  )
  which <- which.matrix(object$which, object$n.vars)
  keep <- apply(which, 1, FUN = function(x) {
    check.heredity(model = x, parents = parents)
  })
  #    priorprobs = apply(which, 1,
  #                 FUN=function(x) {prob.heredity(model=x, parents=parents)}
  #    )
  #    keep = priorprobs > 0.0
  object$n.models <- sum(keep)
  object$sampleprobs <- object$sampleprobs[keep] # if method=MCMC ??? reweight
  object$which <- object$which[keep]
  object$priorprobs <- object$priorprobs[keep] / sum(object$priorprobs[keep])
  #    wts = priorprobs[keep]/object$priorprobs[keep]  #importance weights
  wts <- 1
  method <- object$call$method
  if (!is.null(method)) {
    if (method == "MCMC" || method == "MCMC_new") {
      object$freq <- object$freq[keep]
      #      object$postprobs.MCMC = object$freq[keep]*wts
      object$postprobs.MCMC <- object$freq[keep]
      object$postprobs.MCMC <- object$postprobs.MCMC /
        sum(object$postprobs.MCMC)
      object$probne0.MCMC <- as.vector(object$postprobs.MCMC %*% which[keep, ])
    }
  }
  object$logmarg <- object$logmarg[keep]
  object$shrinkage <- object$shrinkage[keep]
  postprobs.RN <- exp(object$logmarg - min(object$logmarg)) * object$priorprobs
  object$postprobs.RN <- postprobs.RN / sum(postprobs.RN)
  #  browser()
  object$probne0.RN <- as.vector(object$postprobs.RN %*% which[keep, ])

  object$postprobs <- object$postprobs[keep] *
    wts /
    sum(object$postprobs[keep] * wts)
  object$probne0 <- as.vector(object$postprobs %*% which[keep, ])

  object$mle <- object$mle[keep]
  object$mle.se <- object$mle.se[keep]
  object$mse <- object$mse[keep]
  object$size <- object$size[keep]
  object$R2 <- object$R2[keep]
  object$df <- object$df[keep]

  return(object)
}