maxent.jar.predict <- function(mod, envs, other.settings) {
  output.format <- paste0("outputformat=", other.settings$pred.type)
  model.clamp <- ifelse(
    other.settings$doClamp,
    "doclamp=true",
    "doclamp=false"
  )
  pred <- predicts::predict(mod, envs, args = c(output.format, model.clamp))
  return(pred)
}