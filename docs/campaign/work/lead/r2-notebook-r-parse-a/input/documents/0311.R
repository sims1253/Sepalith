mi.combine <- function(x, conf.int = FALSE, conf.level = 0.95) {
  if (requireNamespace("broom", quietly = TRUE)) {
    tidiers <- grep("^tidy\\.", ls(getNamespace("broom")), value = TRUE)
    tidiers <- gsub("tidy\\.", "", tidiers)
  } else {
    rlang::abort("{broom} package required for mi.combine")
  }
  if (!(class(x[[1L]]) %in% tidiers)) {
    rlang::abort("analysis model does not have tidy() method.")
  }

  mi_tidy <- lapply(x, function(x) broom::tidy(x))
  m <- length(mi_tidy)

  out <- mi_tidy[[1L]]
  ests <- est.matrix(mi_tidy, "estimate")
  ses <- est.matrix(mi_tidy, "std.error")
  wi.var <- rowMeans(ses^2)
  out$estimate <- rowMeans(ests)
  diffs <- sweep(ests, 1, rowMeans(ests))
  bw.var <- rowSums(diffs^2) / (m - 1)

  out$std.error <- sqrt(wi.var + bw.var * (1 + 1 / m))
  r <- ((1 + 1 / m) * bw.var) / wi.var
  df <- (m - 1) * (1 + 1 / r)^2
  miss.info <- (r + 2 / (df + 3)) / (r + 1)

  out$statistic <- out$estimate / out$std.error
  out$p.value <- 2 * stats::pt(out$statistic, df = df, lower.tail = FALSE)

  out$df <- df
  out$r <- r
  out$miss.info <- miss.info
  if (conf.int) {
    t.c <- stats::qt(1 - (1 - conf.level) / 2, df = df, lower.tail = FALSE)
    out$conf.low <- out$estimate - t.c * out$std.error
    out$conf.high <- out$estimate + t.c * out$std.error
  }
  out
}