get_exported_functions <- function(
  package,
  ignore_names = "",
  ignore_deprecated = TRUE
) {
  from <- "get_exported_functions"
  validate_class(
    package,
    "character",
    from = from,
    scalar = TRUE,
    remove_empty = TRUE
  )
  validate_class(ignore_names, "character", from = from)
  validate_class(ignore_deprecated, "logical", from = from, scalar = TRUE)
  funs <- tryCatch(sort(getNamespaceExports(package)), error = function(e) {
    fuzz_error(e$message, from = from)
  })

  ## keep only fuzzable functions
  keep.idx <- sapply(funs, function(x) {
    is.function(check_fuzzable(x, package, ignore_deprecated))
  })
  funs <- setdiff(funs[keep.idx], ignore_names)
  attr(funs, "package") <- package
  funs
}