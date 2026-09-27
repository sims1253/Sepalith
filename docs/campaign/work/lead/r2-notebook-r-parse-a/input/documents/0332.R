shortenModelNames <- function(model_names, pad_string = FALSE) {
  model_names_out <- model_names |>
    gsub("exponential", "exp", x = _) |>
    gsub("quadratic", "quad", x = _) |>
    gsub("linear", "lin", x = _) |>
    gsub("logistic", "log", x = _) |>
    gsub("sigEmax", "sigE", x = _) |>
    gsub("betaMod", "betaM", x = _) |>
    gsub("quadratic", "quad", x = _)

  if (pad_string) {
    model_names_out <- padStrings(model_names_out)
  }

  return(model_names_out)
}