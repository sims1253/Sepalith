profile_data <- function(data, by = NULL, alpha = 0.05, digits = NULL) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame or tibble.", call. = FALSE)
  }

  summary <- summarize_data(data, by = by, alpha = alpha, digits = digits)
  type_values <- vapply(data, .column_type, character(1))
  type_counts <- table(type_values)
  total_cells <- nrow(data) * ncol(data)
  total_missing <- sum(vapply(data, function(column) sum(.missing_index(column)), integer(1)))

  dataset <- data.frame(
    rows = nrow(data),
    columns = ncol(data),
    complete_rows = sum(stats::complete.cases(data)),
    duplicated_rows = sum(duplicated(data)),
    total_missing = total_missing,
    missing_pct = if (total_cells == 0) NA_real_ else 100 * total_missing / total_cells,
    type_profile = paste(names(type_counts), as.integer(type_counts), sep = "=", collapse = ", "),
    stringsAsFactors = FALSE
  )
  dataset <- .round_numeric_columns(dataset, digits)

  profile <- list(
    generated_at = Sys.time(),
    alpha = alpha,
    dataset = dataset,
    summary = summary,
    warnings = .profile_warnings(summary, dataset)
  )
  class(profile) <- "datasum_profile"
  profile
}