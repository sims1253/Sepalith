p.heatmap <- function(
  x,
  adj.method = "none",
  only.sig = TRUE,
  cutoff = ifelse(adj.method == "none", 0.01, 0.05)
) {
  if (!inherits(x, "cepa.all")) {
    stop("x should be cepa.all object.\n")
  }

  p.value <- p.table(x)
  p.foo <- apply(p.value, 2, p.adjust, adj.method)
  if (!is.matrix(p.foo)) {
    p.foo <- matrix(p.foo, nrow = 1)
    rownames(p.foo) <- rownames(p.value)
    colnames(p.foo) <- colnames(p.value)
    p.value <- p.foo
  }
  p.value <- p.foo

  np <- ncol(p.value)

  if (only.sig) {
    l <- apply(p.value, 1, function(x) {
      sum(x <= cutoff) > 0
    })
    p.value <- p.value[l, , drop = FALSE]
    if (sum(l) == 0) {
      plot(1, 1, axes = FALSE, ann = FALSE, type = "n")
      text(1, 1, "No significant pathway!", cex = 2)
      return(invisible(NULL))
    }
  }

  # get the order of p.fisher
  o <- order(rowMeans(-log(p.value + 1e-8)), decreasing = TRUE)

  p.smallest <- min(p.value) + 1e-4
  p.value[p.value == 0] <- p.smallest
  p2 <- -log10(p.value)
  p2 <- p2[o, , drop = FALSE]

  # smallest p value is 0.001
  #p2[p2 > floor(-log10(p.smallest))] = floor(-log10(p.smallest))

  p2 <- t(p2)

  if (only.sig) {
    # do not draw pathway names
    cn <- colnames(p2)
    max.length <- max(nchar(cn))
    layout(
      rbind(c(1, 2), c(3, 4), c(5, 0)),
      widths = c(7, lcm(3.5)),
      heights = c(1.5, 5, max.length / 4)
    )
  } else {
    layout(
      rbind(c(1, 2), c(3, 4)),
      widths = c(7, lcm(3.5)),
      heights = c(1.5, 5)
    )
  }

  # 1.only draw titles
  par("mar" = c(0, 0, 0, 0))
  plot(0, 0, type = "n", axes = FALSE, ann = FALSE)
  if (only.sig) {
    text(
      0,
      0,
      paste(
        "Heatmap of ",
        ifelse(adj.method == "none", "p-value", "FDR"),
        "s of pathways (only significant)",
        sep = ""
      ),
      cex = 1.5
    )
  } else {
    text(
      0,
      0,
      paste(
        "Heatmap of ",
        ifelse(adj.method == "none", "p-value", "FDR"),
        "s of pathways",
        sep = ""
      ),
      cex = 1.5
    )
  }

  fc <- c(0, -log10(cutoff), floor(-log10(p.smallest)) + 1)
  colors <- c("green", "white", "red")

  # 2. legend
  g <- seq(0, floor(-log10(p.smallest)) + 1, length.out = 100)
  lg <- length(g)
  plot(
    c(0, 0),
    c(lg, 3),
    type = "n",
    xlim = c(0, lg * 1.1),
    ylim = c(0, 3),
    axes = FALSE,
    ann = FALSE
  )
  text(1, 1.5, 1)
  text(lg, 1.5, 10^(-fc[3]))
  text((fc[2] - fc[1]) / (fc[3] - fc[1]) * (lg - 1) + 1, 1.5, 10^(-fc[2]))
  for (i in 1:lg) {
    rect(
      i - 1,
      0,
      i,
      1,
      col = get_color(g[i], colors = colors, fc = fc),
      border = NA
    )
  }

  # 3. heatmap
  par("mar" = c(0, 0, 0, 0))
  ncol <- dim(p2)[2]
  nrow <- dim(p2)[1]
  plot(
    c(0, 0),
    c(ncol, nrow),
    type = "n",
    xlim = c(0, ncol),
    ylim = c(0, nrow),
    axes = FALSE,
    ann = FALSE,
    xaxs = "i"
  )

  for (i in 1:nrow) {
    for (j in 1:ncol) {
      rect(
        j - 1,
        i - 1,
        j,
        i,
        col = get_color(p2[i, j], colors = colors, fc = fc),
        border = NA
      )
    }
  }

  # 4. rownames
  rn <- rownames(p2)
  par("mar" = c(0, 0, 0, 2))
  plot(
    c(0, 0),
    c(1, nrow),
    type = "n",
    xlim = c(0, 1),
    ylim = c(0, nrow),
    axes = FALSE,
    ann = FALSE
  )
  for (i in 1:nrow) {
    text(0, i - 0.5, rn[i], adj = c(0, 0.5), cex = ifelse(only.sig, 1.5, 1.2))
  }

  # 5. colnames
  if (only.sig) {
    cn <- colnames(p2)
    par("mar" = c(2, 0, 0, 0))
    plot(
      c(0, 0),
      c(ncol, 1),
      type = "n",
      xlim = c(0, ncol),
      ylim = c(0, 1),
      axes = FALSE,
      ann = FALSE,
      xaxs = "i"
    )
    for (i in 1:ncol) {
      text(i - 0.5, 1, cn[i], srt = -90, adj = c(0, 0.5), cex = 1.5)
    }
  }

  par("mar" = c(5.1, 4.1, 4.1, 2.1))
  layout(matrix(1, 1, 1))
  return(invisible(NULL))
}