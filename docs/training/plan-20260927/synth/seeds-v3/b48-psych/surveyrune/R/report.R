#' Format scores for a report
#'
#' @description
#' Turns numeric scores into display strings for one score type.
#'
#' @param x A numeric vector of scores.
#' @param type One of `"raw"`, `"z"`, `"t"`, `"iq"`, `"percentile"` or
#'   `"stanine"`.
#' @param digits Number of decimals for `"raw"` and `"z"`.
#' @return A character vector the length of `x`, with `"--"` for missing
#'   scores.
#' @export
format.score <- function(x, type = c("raw", "z", "t", "iq", "percentile", "stanine"),
                         digits = 1) {
  type <- match.arg(type)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }

  out <- switch(type,
    raw = {
      formatC(x, format = "f", digits = digits)
    },
    z = {
      sprintf("%+.*f", digits, x)
    },
    t = {
      sprintf("T = %d", as.integer(round(x)))
    },
    iq = {
      sprintf("IQ = %d", as.integer(round(x)))
    },
    percentile = {
      p <- pmin(pmax(round(x), 1), 99)
      sprintf("PR = %d", as.integer(p))
    },
    stanine = {
      s <- pmin(pmax(round(x), 1), 9)
      sprintf("Stanine %d", as.integer(s))
    }
  )
  out[is.na(x)] <- "--"
  out
}

#' Label T scores with descriptive bands
#'
#' @description
#' Maps T scores to the labels of one banding scheme.
#'
#' @param t A numeric vector of T scores.
#' @param scheme One of `"five"`, `"three"`, `"clinical"` or `"strength"`.
#' @return A character vector of band labels.
#' @export
band.label <- function(t, scheme = c("five", "three", "clinical", "strength")) {
  scheme <- match.arg(scheme)

  bands <- switch(scheme,
    five = {
      breaks <- c(-Inf, 30, 40, 60, 70, Inf)
      labels <- c("very low", "low", "average", "high", "very high")
      list(breaks = breaks, labels = labels)
    },
    three = {
      breaks <- c(-Inf, 40, 60, Inf)
      labels <- c("below average", "average", "above average")
      list(breaks = breaks, labels = labels)
    },
    clinical = {
      breaks <- c(-Inf, 60, 65, 70, Inf)
      labels <- c("normal", "elevated", "borderline", "clinical")
      list(breaks = breaks, labels = labels)
    },
    strength = {
      breaks <- c(-Inf, 40, 55, Inf)
      labels <- c("development area", "typical", "strength")
      list(breaks = breaks, labels = labels)
    }
  )
  as.character(cut(t, bands$breaks, labels = bands$labels, right = FALSE))
}

#' Build a score profile
#'
#' @description
#' Converts raw subscale scores of one respondent to z, T and percentile
#' scores with a band label.
#'
#' @param raw A named numeric vector of raw subscale scores.
#' @param means,sds Named numeric vectors of norm means and standard
#'   deviations, with the same names as `raw`.
#' @return A data frame with columns `scale`, `raw`, `z`, `t`, `percentile`
#'   and `band`, one row per subscale.
#' @export
score.profile <- function(raw, means, sds) {
  scales <- names(raw)
  if (is.null(scales) || !all(scales %in% names(means)) || !all(scales %in% names(sds))) {
    stop("`raw`, `means` and `sds` must share the same names.", call. = FALSE)
  }
  out <- data.frame(scale = scales, raw = unname(raw))

  # 1. z scores
  out$z <- (out$raw - unname(means[scales])) / unname(sds[scales])
  # 2. T scores
  out$t <- round(50 + 10 * out$z)
  # 3. Percentiles
  out$percentile <- round(100 * stats::pnorm(out$z))
  # 4. Bands
  out$band <- band.label(out$t)

  out
}

#' Describe a score profile in sentences
#'
#' @description
#' Writes one sentence per subscale from a profile.
#'
#' @param profile A data frame as returned by [score.profile()].
#' @param name The respondent's name or code.
#' @return A character vector, one sentence per subscale.
#' @export
profile.text <- function(profile, name = "The respondent") {
  needed <- c("scale", "t", "percentile", "band")
  if (!all(needed %in% names(profile))) {
    stop("`profile` must have columns `scale`, `t`, `percentile` and `band`.", call. = FALSE)
  }

  sentences <- character(nrow(profile))
  for (i in seq_len(nrow(profile))) {
    row <- profile[i, ]
    if (is.na(row$t)) {
      sentences[i] <- sprintf("%s has no valid score on %s.", name, row$scale)
      next
    }
    sentences[i] <- sprintf(
      "%s scored in the %s range on %s (T = %d, percentile %d).",
      name, row$band, row$scale, as.integer(row$t), as.integer(row$percentile)
    )
  }
  sentences
}

#' Export a score profile
#'
#' @description
#' Writes a profile table as CSV, Markdown or HTML.
#'
#' @param profile A data frame as returned by [score.profile()].
#' @param path Output file path.
#' @param format One of `"csv"`, `"markdown"` or `"html"`.
#' @return `path`, invisibly.
#' @export
export.profile <- function(profile, path, format = c("csv", "markdown", "html")) {
  format <- match.arg(format)
  cells <- as.matrix(format(profile))

  lines <- switch(format,
    csv = {
      utils::write.csv(profile, path, row.names = FALSE)
      NULL
    },
    markdown = {
      header <- paste0("| ", paste(names(profile), collapse = " | "), " |")
      rule <- paste0("|", paste(rep("---", ncol(profile)), collapse = "|"), "|")
      body <- apply(cells, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
      c(header, rule, body)
    },
    html = {
      header <- paste0("<tr>", paste0("<th>", names(profile), "</th>", collapse = ""), "</tr>")
      body <- apply(cells, 1, function(r) {
        paste0("<tr>", paste0("<td>", r, "</td>", collapse = ""), "</tr>")
      })
      c("<table>", header, body, "</table>")
    }
  )
  if (!is.null(lines)) {
    writeLines(lines, path)
  }
  invisible(path)
}

#' Plot a score profile
#'
#' @description
#' Draws T scores per subscale against the average band.
#'
#' @param profile A data frame as returned by [score.profile()].
#' @param title Plot title.
#' @return A ggplot object.
#' @importFrom ggplot2 ggplot aes
#' @export
profile.plot <- function(profile, title = "Score profile") {
  profile$scale <- factor(profile$scale, levels = profile$scale)

  ggplot2::ggplot(profile, ggplot2::aes(x = scale, y = t, group = 1)) +
    ggplot2::annotate("rect", xmin = -Inf, xmax = Inf, ymin = 40, ymax = 60,
      alpha = 0.15, fill = "grey50") +
    ggplot2::geom_hline(yintercept = 50, linetype = "dashed") +
    ggplot2::geom_line() +
    ggplot2::geom_point(ggplot2::aes(colour = band), size = 3) +
    ggplot2::scale_y_continuous(limits = c(20, 80), breaks = seq(20, 80, by = 10)) +
    ggplot2::labs(title = title, x = NULL, y = "T score", colour = "Band") +
    ggplot2::theme_minimal()
}
