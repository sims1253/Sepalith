prob.heredity <- function(model, parents, prob = 0.5) {
  got.parents <- apply(parents, 1, FUN = function(x) {
    all(as.logical(model[as.logical(x)]))
  })
  model.prob <- 0
  if (all(model == got.parents)) {
    model.prob <- exp(
      sum(model * log(prob) + (1 - model) * log(1.0 - prob))
    )
  }
  return(model.prob)
}