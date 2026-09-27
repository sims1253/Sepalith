.extractCol2 <- function(contrastList, colName) {
  # Support combining topTable data from different DGEobjs
  assertthat::assert_that(
    "list" %in% class(contrastList),
    !is.null(names(contrastList)),
    msg = "contrastList must be a named list."
  )
  assertthat::assert_that(
    "character" %in% class(colName),
    msg = "colName must be a column in the data of class 'character'."
  )

  for (i in seq_along(contrastList)) {
    newdat <- cbind(
      rowid = rownames(contrastList[[i]]),
      data.frame(contrastList[[i]], row.names = NULL)
    )
    newdat <- newdat[, c("rowid", colName)]

    if (i == 1) {
      dat <- newdat
    } else {
      dat <- merge(x = dat, y = newdat, by = "rowid", all = TRUE, sort = FALSE)
    }
    colnames(dat)[i + 1] <- names(contrastList[i])
  }
  dat <- data.frame(dat[, -c(1)], row.names = dat[, c(1)])
  return(dat)
}