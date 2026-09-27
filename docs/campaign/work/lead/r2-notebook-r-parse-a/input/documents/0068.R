summary.boot_RRi_fit <- function(object, robust = TRUE, ...) {
  dt <- data.table::as.data.table(object)
  # Exclude the replicate identifier "nboot" from the parameters
  summary_list <- dt[,
    lapply(.SD, function(x) {
      if (isTRUE(robust)) {
        estimate <- round(stats::median(x), 2)
        scale <- round(stats::mad(x), 2)
      } else {
        estimate <- round(mean(x), 2)
        scale <- round(stats::sd(x), 2)
      }
      Q2.5 <- round(stats::quantile(x, probs = 0.025), 2)
      Q97.5 <- round(stats::quantile(x, probs = 0.975), 2)
      c(
        estimate,
        scale,
        paste0("[", Q2.5, ", ", Q97.5, "]")
      )
    }),
    .SDcols = -c("nboot")
  ]

  summary_list[, `:=`(
    Parameter = c("Estimate", "SE", "95% CI")
  )]

  summary_list <- data.table::transpose(
    summary_list,
    keep.names = "Parameter",
    make.names = "Parameter"
  )

  class(summary_list) <- c("summary.boot_RRi_fit", class(summary_list))
  attr(summary_list, which = "robust") <- robust
  return(summary_list)
}