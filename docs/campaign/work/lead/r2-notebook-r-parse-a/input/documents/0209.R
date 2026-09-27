monitor_arrange <- function(
  monitor,
  ...
) {
  # ----- Validate parameters --------------------------------------------------

  # A little involved to catch the case where the user forgets to pass in 'monitor'

  result <- try(
    {
      if (!monitor_isValid(monitor)) {
        stop("First argument is not a valid 'mts_monitor' object.")
      }
    },
    silent = TRUE
  )

  if (inherits(result, "try-error")) {
    err_msg <- geterrmessage()
    if (stringr::str_detect(err_msg, "object .* not found")) {
      stop(paste0(
        err_msg,
        "\n(Did you forget to pass in the 'monitor' object?)"
      ))
    }
  }

  # ----- Call MazamaTimeSeries function ---------------------------------------

  monitor <- MazamaTimeSeries::mts_arrange(monitor, ...)
  class(monitor) <- union("mts_monitor", class(monitor))

  # ----- Return ---------------------------------------------------------------

  return(monitor)
}