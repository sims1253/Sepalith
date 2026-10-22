#' Format counts and percentages as "n (x.x%)"
#'
#' @param n Integer vector of participant counts.
#' @param pct Numeric vector of percentages.
#' @param digits Number of decimal places for percentages.
#' @return A character vector.
#' @keywords internal
fmt.n.pct = function(n, pct, digits = 1) {
  out = sprintf(paste0("%d (%.", digits, "f%%)"), as.integer(n), pct)
  out[n == 0] = "0"
  out
}

#' Format a header label with the arm size
#'
#' @param arm Character vector of arm labels.
#' @param n.arm Integer vector of participants per arm.
#' @return A character vector such as `"Placebo (N=120)"`.
#' @keywords internal
fmt.arm.header = function(arm, n.arm) {
  sprintf("%s (N=%d)", arm, as.integer(n.arm))
}

#' Spread an adverse event count table to one column per arm
#'
#' Turns the long output of the `ae_by_*()` helpers into the familiar wide
#' layout, with one formatted "n (%)" column per arm.
#'
#' @param counts A long count table with columns `arm`, `n`, `n_arm` and `pct`.
#' @param digits Number of decimal places for percentages.
#' @return A data frame with the term columns followed by one column per arm.
#' @family ae table builders
#' @export
ae.table.wide = function(counts, digits = 1) {
  counts = as.data.frame(counts)
  keys = setdiff(names(counts), c("arm", "n", "n_arm", "pct"))
  arms = unique(counts[, c("arm", "n_arm")])
  arms = arms[order(arms$arm), , drop = FALSE]
  counts$cell = fmt.n.pct(counts$n, counts$pct, digits = digits)
  counts$key = do.call(paste, c(counts[keys], sep = "||"))
  out = unique(counts[c("key", keys)])
  for (i in seq_len(nrow(arms))) {
    sub = counts[counts$arm == arms$arm[i], ]
    col = fmt.arm.header(arms$arm[i], arms$n_arm[i])
    out[[col]] = sub$cell[match(out$key, sub$key)]
    out[[col]][is.na(out[[col]])] = "0"
  }
  out$key = NULL
  rownames(out) = NULL
  out
}

#' Sort an adverse event table
#'
#' Orders rows either by decreasing total frequency across arms or
#' alphabetically by term, keeping preferred terms inside their system
#' organ class.
#'
#' @param counts A long count table with columns `arm`, `n`, `n_arm` and `pct`.
#' @param by Sort order, `"frequency"` or `"alphabetical"`.
#' @return `counts`, reordered.
#' @family ae table builders
#' @export
ae.table.sort = function(counts, by = c("frequency", "alphabetical")) {
  by = match.arg(by)
  counts = as.data.frame(counts)
  has.pt = "aedecod" %in% names(counts)
  term = if (has.pt) counts$aedecod else counts$aesoc
  ord = switch(by,
    frequency = {
      soc.total = ave(counts$n, counts$aesoc, FUN = sum)
      term.total = ave(counts$n, counts$aesoc, term, FUN = sum)
      order(-soc.total, counts$aesoc, -term.total, term, counts$arm)
    },
    alphabetical = order(counts$aesoc, term, counts$arm)
  )
  counts[ord, , drop = FALSE]
}

#' Keep terms reported above a frequency threshold
#'
#' Retains every row of a term when its incidence reaches `min.pct` in at
#' least one arm, the usual rule for "common adverse events" tables.
#'
#' @param counts A long count table with columns `arm`, `n`, `n_arm` and `pct`.
#' @param min.pct Minimum incidence, in percent, in any arm.
#' @return `counts`, filtered.
#' @family ae table builders
#' @export
ae.table.common = function(counts, min.pct = 5) {
  counts = as.data.frame(counts)
  keys = setdiff(names(counts), c("arm", "n", "n_arm", "pct"))
  key = do.call(paste, c(counts[keys], sep = "||"))
  max.pct = ave(counts$pct, key, FUN = max)
  counts[max.pct >= min.pct, , drop = FALSE]
}

#' Listing of serious adverse events
#'
#' Selects serious events and orders them by participant and onset date for
#' the narrative listing appendix.
#'
#' @param ae An event-level adverse event data frame, one row per event.
#' @param cols Columns to keep in the listing, in display order.
#' @return A data frame with one row per serious event.
#' @export
ae.listing.serious = function(ae, cols = c("usubjid", "trt01a", "aedecod", "aestdt",
                                           "aeendt", "aesev", "aerel", "aeout")) {
  missing.cols = setdiff(cols, names(ae))
  if (length(missing.cols) > 0) {
    stop("Columns not in `ae`: ", paste(missing.cols, collapse = ", "), call. = FALSE)
  }
  ser = ae[!is.na(ae$aeser) & ae$aeser == "Y", , drop = FALSE]
  ser = ser[order(ser$usubjid, ser$aestdt), cols, drop = FALSE]
  if ("aeendt" %in% cols && "aestdt" %in% cols) {
    ser$duration.days = as.integer(ser$aeendt - ser$aestdt) + 1L
  }
  rownames(ser) = NULL
  ser
}

#' Write an adverse event table to disk
#'
#' Writes a formatted table as CSV, tab-separated text or a Markdown pipe
#' table for quick review.
#'
#' @param tab A formatted table, e.g. from [ae.table.wide()].
#' @param file Path of the file to write.
#' @param format Output format, one of `"csv"`, `"tsv"` or `"markdown"`.
#' @return `file`, invisibly.
#' @export
write.ae.table = function(tab, file, format = c("csv", "tsv", "markdown")) {
  format = match.arg(format)
  switch(format,
    csv = utils::write.csv(tab, file, row.names = FALSE),
    tsv = utils::write.table(tab, file, sep = "\t", quote = FALSE, row.names = FALSE),
    markdown = {
      header = paste0("| ", paste(names(tab), collapse = " | "), " |")
      rule = paste0("|", paste(rep("---", ncol(tab)), collapse = "|"), "|")
      body = apply(tab, 1, function(row) paste0("| ", paste(row, collapse = " | "), " |"))
      writeLines(c(header, rule, body), file)
    }
  )
  invisible(file)
}
