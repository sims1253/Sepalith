sanitize_names <- function(variables_names) {
  sanitized_names <- c()
  for (name in variables_names) {
    if (grepl('-', name, fixed = TRUE)) {
      original_name <- name
      name <- gsub('-', '_', name)
      if (name %in% variables_names) {
        stop(paste(
          'The group names contain minus characters (-) which prevent intersections names composition;',
          'offending group:',
          original_name,
          'please substitute these characters using gsub and try again.'
        ))
      }
    }
    sanitized_names <- c(sanitized_names, name)
  }
  sanitized_names
}