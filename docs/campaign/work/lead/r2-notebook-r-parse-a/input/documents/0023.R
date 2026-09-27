Simplify.abs <- function(expr, scache = NULL) {
  if (is.uminus(expr[[2]])) {
    expr[[2]] <- expr[[2]][[2]]
  } else if (is.call(expr[[2]])) {
    subf <- format1(expr[[2]][[1]])
    if (subf == "^") {
      p <- expr[[2]][[3]]
      if (is.numeric(p) && p %% 2 == 0) {
        expr <- expr[[2]]
      }
    } else if (subf == "exp" || subf == "sqrt") {
      expr <- expr[[2]]
    }
  }
  expr
}