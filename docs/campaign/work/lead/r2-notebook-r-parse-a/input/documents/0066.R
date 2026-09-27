r.pval <- function(r, n, h0, alternative, ROPE = NULL) {
  Z.obs <- r_mean(r)
  Z.sd <- r_sd(n)
  Z.h0 <- r_mean(h0)
  if (is.null(ROPE)) {
    p <- switch(
      alternative,
      "less" = stats::pnorm(Z.obs, mean = Z.h0, sd = Z.sd),
      "greater" = stats::pnorm(Z.obs, mean = Z.h0, sd = Z.sd, lower = F),
      "two.sided" = min(
        stats::pnorm(Z.obs, mean = Z.h0, sd = Z.sd, lower = F),
        stats::pnorm(Z.obs, mean = Z.h0, sd = Z.sd, lower = T)
      ) *
        2
    )
  } else {
    ROPE <- h0 + ROPE

    Z.h1 <- Z.h0 + r_mean(ROPE)

    p <- switch(
      alternative,
      "less" = stats::pnorm(Z.obs, mean = Z.h1, sd = Z.sd, lower.tail = F),
      "greater" = stats::pnorm(Z.obs, mean = Z.h1, sd = Z.sd, lower.tail = T),
      "two.sided" = max(
        stats::pnorm(Z.obs, mean = max(Z.h1), sd = Z.sd, lower.tail = T),
        stats::pnorm(Z.obs, mean = min(Z.h1), sd = Z.sd, lower.tail = F)
      )
    )
  }

  return(p)
}