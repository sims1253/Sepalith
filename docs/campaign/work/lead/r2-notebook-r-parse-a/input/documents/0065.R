t.pval <- function(
  tval,
  n1,
  n2 = NULL,
  alternative,
  ROPE = NULL,
  type = "One-sample t-test"
) {
  # Degrees of freedom and scaling constant
  if (type == "One-sample t-test") {
    df <- n1 - 1
    constant <- sqrt(n1)
  } else {
    df <- n1 + n2 - 2
    constant <- sqrt(n1 * n2 / (n1 + n2))
  }

  # No ROPE: standard p-values
  if (is.null(ROPE)) {
    p <- switch(
      alternative,
      "less" = stats::pt(tval, df),
      "greater" = stats::pt(tval, df, lower.tail = FALSE),
      "two.sided" = 1 - (stats::pt(abs(tval), df) - stats::pt(-abs(tval), df))
    )
  } else {
    # ROPE specified
    if (alternative %in% c("less", "greater")) {
      # ROPE must be length 1 for one-sided tests
      ncp <- ROPE * constant
      p <- switch(
        alternative,
        "less" = stats::pt(tval, df, ncp = ncp, lower.tail = FALSE),
        "greater" = stats::pt(tval, df, ncp = ncp, lower.tail = TRUE)
      )
    } else if (alternative == "two.sided") {
      # ROPE must be length 2 for two-sided
      if (length(ROPE) != 2) {
        stop("For alternative '!=', ROPE must be of length 2.")
      }
      ncp_lower <- ROPE[1] * constant
      ncp_upper <- ROPE[2] * constant
      p <- max(
        stats::pt(tval, df, ncp = ncp_lower, lower.tail = FALSE),
        stats::pt(tval, df, ncp = ncp_upper, lower.tail = TRUE)
      )
    } else {
      stop("Invalid alternative. Must be '<', '>', or '!='.")
    }
  }

  return(p)
}