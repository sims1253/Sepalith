dt2_buttons <- function(
  options = list(),
  buttons = c("copyHtml5", "csvHtml5", "excelHtml5", "pdfHtml5", "print"),
  target = NULL
) {
  options$buttons <- as.list(buttons)
  if (!is.null(target)) {
    options$dt2_buttons_target <- target
  }
  options
}