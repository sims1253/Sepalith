.CollapseTable <- function(counts, max.levels, alphabetical = TRUE) {
  ## Truncates a table to have at most max.levels of entries.  The
  ## other entries will be collapsed into a single level called
  ## "other".  If alphabetical == TRUE then the returned table will be
  ## sorted alphabetically according to the table name.  Otherwise it
  ## will be sorted by frequency, in decreasing order.
  if (length(counts) <= max.levels) {
    return(
      if (alphabetical) {
        .OrderAlphabetically(counts)
      } else {
        .OrderByFrequency(counts)
      }
    )
  }
  index <- rev(order(counts))
  new.tab <- counts[index][1:(max.levels - 1)]
  other <- sum(counts[index][-(1:max.levels - 1)])
  if (alphabetical) {
    alpha.index <- order(names(new.tab))
    new.tab <- new.tab[alpha.index]
  }
  new.tab <- c(new.tab, other)

  ## Make sure the last entry has a name, and its name is 'other'
  names(new.tab)[length(new.tab)] <- "other"

  return(new.tab)
}