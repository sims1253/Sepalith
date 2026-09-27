.training_groups <- function(object) {
  y <- object$training$y
  groups <- object$training$stability_groups
  if (is.null(y)) {
    return(character(0))
  }
  if (is.null(groups) || length(groups) != nrow(y)) {
    groups <- rep("All", nrow(y))
  }
  groups <- as.character(groups)
  groups[is.na(groups) | groups == ""] <- "Missing"
  groups
}