maxent.jar.msgs <- function(tune.args, other.settings) {
  rJava::.jpackage("predicts")
  mxe <- rJava::.jnew("meversion")
  v <- try(rJava::.jcall(mxe, "S", "meversion"))
  msg <- paste0(
    "maxent.jar v",
    v,
    " using the predicts package v",
    packageVersion('predicts')
  )
  return(msg)
}