dt.remove.variables <- function(
  dt.name,
  the.variables,
  return.as = "result",
  envir = .GlobalEnv,
  ...
) {
  is.format.dt <- check.dt.status(dt.name = dt.name, envir = envir)

  all.variable.names <- names(x = get(dt.name))

  if (is.numeric(the.variables)) {
    the.indices <- unique(floor(the.variables))

    the.variables <- all.variable.names[the.indices[
      the.indices %in% seq_len(ncol(x = get(x = dt.name, envir = envir)))
    ]]
  }

  the.variables <- the.variables[
    the.variables %in% names(get(x = dt.name, envir = envir))
  ]
  num.variables <- length(the.variables)
  if (num.variables == 0) {
    j.statement <- NULL
  }
  if (num.variables == 1) {
    j.statement <- sprintf("%s := NULL", add.backtick(x = the.variables))
  }
  if (num.variables > 1) {
    j.statement <- sprintf(
      "c(%s) := NULL",
      paste(sprintf("'%s'", the.variables), collapse = ", ")
    )
  }

  the.statement <- create.dt.statement(
    dt.name = dt.name,
    j.statement = j.statement
  )

  res <- eval.dt.statement(
    the.statement = the.statement,
    return.as = return.as,
    envir = .GlobalEnv
  )

  revise.dt.status(
    dt.name = dt.name,
    envir = envir,
    is.format.dt = is.format.dt
  )

  return(res)
}