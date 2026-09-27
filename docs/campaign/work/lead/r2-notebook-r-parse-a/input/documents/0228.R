validate_engine_args_container <- function(engine.args) {
  if (is.null(engine.args)) {
    return(list())
  }
  if (!is.list(engine.args)) {
    stop("`engine.args` must be NULL or a named list.", call. = FALSE)
  }
  if (!length(engine.args)) {
    return(list())
  }

  nm <- names(engine.args)
  if (is.null(nm)) {
    stop("`engine.args` must be a named list.", call. = FALSE)
  }
  if (any(is.na(nm) | !nzchar(nm))) {
    stop("`engine.args` contains missing or empty names.", call. = FALSE)
  }
  if (anyDuplicated(nm)) {
    dup <- unique(nm[duplicated(nm)])
    stop(
      "`engine.args` contains duplicated names: ",
      paste(dup, collapse = ", "),
      call. = FALSE
    )
  }

  engine.args
}