all.amr_selector_any_all <- function(..., na.rm = FALSE) {
  # this is all() on a logical vector from `==.amr_selector` or `!=.amr_selector`
  # e.g., example_isolates %>% filter(all(carbapenems() == "R"))
  # so just return the vector as is, only correcting for na.rm
  out <- unclass(c(...))
  if (isTRUE(na.rm)) {
    out <- out[!is.na(out)]
  }
  out
}