.coerce_numeric_cols <- function(df) {
  for (col in names(df)) {
    x <- df[[col]]
    if (is.character(x)) {
      num <- suppressWarnings(as.numeric(x))
      if (!anyNA(num[!is.na(x) & nzchar(x)]) || all(is.na(x) | !nzchar(x))) {
        # only coerce if every non-empty value parsed successfully
        if (!any(is.na(num) & !is.na(x) & nzchar(x))) df[[col]] <- num
      }
    }
  }
  df
}