#' Setup Bigsimr
#'
#' @param pkg_check If TRUE, checks if Bigsimr is installed and installs it if
#'   necessary.
#' @param ... Additional arguments passed to `julia_setup()`.
#'
#' @return A list of Bigsimr functions.
#'
#' @export
bigsimr_setup <- function (pkg_check = TRUE, ...) {
  julia <- JuliaCall::julia_setup(installJulia = TRUE, ...)

  if (pkg_check) {
    JuliaCall::julia_install_package_if_needed("Bigsimr")
  }

  JuliaCall::julia_library("Bigsimr")

  functions <- JuliaCall::julia_eval(
    "filter(isascii, string.(propertynames(Bigsimr)))"
  )

  rm_funcs <- c("Bigsimr",
                "NearestCorrelationMatrix",
                "PearsonCorrelationMatch",
                "CorType",
                "Pearson",
                "Spearman",
                "Kendall")

  functions <- functions[!(functions %in% rm_funcs)]

  bs <- JuliaCall::julia_pkg_import("Bigsimr", functions)
  bs$Pearson  <- JuliaCall::julia_eval("Pearson")
  bs$Spearman <- JuliaCall::julia_eval("Spearman")
  bs$Kendall  <- JuliaCall::julia_eval("Kendall")

  bs
}