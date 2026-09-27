check_algo_options_woa <- function(p, ...) {
  config_options <- list(...)
  if (length(config_options) == 0) {
    return(p)
  }
  for (i in seq_along(config_options)) {
    stop(paste0(
      "Unknown option '",
      names(config_options[i]),
      "' for algorithm WOA."
    ))
  }
  return(p)
}