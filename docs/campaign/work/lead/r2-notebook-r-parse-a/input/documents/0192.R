OneHotEncoder.fit <- function(X) {
  if (
    !is.matrix(X) &&
      !is.data.frame(X)
  ) {
    stop("input can only be matrix or data.frame")
  }
  if (is.data.table(X)) {
    column_encoders <-
      sapply(seq_len(ncol(X)), function(i) {
        LabelEncoder.fit(X[, i, with = FALSE][[1]])
      })
  } else {
    column_encoders <-
      sapply(seq_len(ncol(X)), function(i) {
        LabelEncoder.fit(X[, i])
      })
  }
  n_values <-
    unlist(lapply(column_encoders, function(e) {
      length(e@classes)
    }))
  encoder <- new(
    "OneHotEncoder",
    n_columns = ncol(X),
    n_values = n_values,
    column_encoders = column_encoders
  )
  return(encoder)
}