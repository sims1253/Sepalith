LabelEncoder.fit <- function(y) {
  if (
    !is.numeric(y) &&
      !is.character(y) &&
      !is.factor(y)
  ) {
    stop("input can only be either numeric or character or factor")
  }
  classes <- sort(unique(y), na.last = TRUE)
  map <- data.table::data.table(classes)
  data.table::setkey(map, "classes")
  ind <- 0L # prevent a warning when build package
  map[, ind := seq_along(classes)]
  if (is.numeric(y)) {
    encoder <- new(
      "LabelEncoder.Numeric",
      type = "numeric",
      classes = classes,
      mapping = as.data.frame(map)
    )
  } else if (is.character(y)) {
    encoder <- new(
      "LabelEncoder.Character",
      type = "character",
      classes = classes,
      mapping = as.data.frame(map)
    )
  } else {
    encoder <- new(
      "LabelEncoder.Factor",
      type = "factor",
      classes = classes,
      mapping = as.data.frame(map)
    )
  }
  return(encoder)
}