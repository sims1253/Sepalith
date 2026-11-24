#' Screen questionnaire responses for quality problems
#'
#' @description
#' Runs the range, long-string, missing and speed checks and flags each
#' respondent.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items Character vector naming the items to screen.
#' @param min.value,max.value The lowest and highest valid response option.
#' @param max.longstring Longest allowed run of identical answers.
#' @param max.missing Largest allowed share of missing items.
#' @param seconds Optional numeric vector of completion times in seconds.
#' @param min.seconds Shortest plausible completion time.
#' @return A data frame with columns `out.of.range`, `longstring`, `missing`,
#'   `speeder` and `n.flags`.
#' @export
screen.responses <- function(data, items, min.value = 1, max.value = 5,
                             max.longstring = 10, max.missing = 0.2,
                             seconds = NULL, min.seconds = 60) {
  responses <- as.matrix(data[items])
  out <- data.frame(row.names = seq_len(nrow(responses)))

  # Check `out.of.range`
  bad <- !is.na(responses) & (responses < min.value | responses > max.value)
  out$out.of.range <- rowSums(bad) > 0

  # Check `longstring`
  run.length <- apply(responses, 1, function(row) {
    runs <- rle(as.character(row[!is.na(row)]))
    if (length(runs$lengths) == 0) 0L else max(runs$lengths)
  })
  out$longstring <- run.length > max.longstring

  # Check `missing`
  share.missing <- rowMeans(is.na(responses))
  out$missing <- share.missing > max.missing

  # Check `speeder`
  if (is.null(seconds)) {
    out$speeder <- FALSE
  } else {
    if (length(seconds) != nrow(responses)) {
      stop("`seconds` must have one value per respondent.", call. = FALSE)
    }
    out$speeder <- !is.na(seconds) & seconds < min.seconds
  }

  out$n.flags <- rowSums(out[c("out.of.range", "longstring", "missing", "speeder")])
  out
}

#' Careless-responding index
#'
#' @description
#' Computes one careless-responding index per respondent.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param items Character vector naming the items to screen.
#' @param index One of `"longstring"`, `"irv"`, `"mahalanobis"`,
#'   `"even.odd"` or `"span"`.
#' @param scales For `"even.odd"`, a list of character vectors naming the
#'   items of each scale.
#' @return A numeric vector, one value per respondent, with attribute `index`.
#' @export
careless.index <- function(data, items,
                           index = c("longstring", "irv", "mahalanobis", "even.odd", "span"),
                           scales = NULL) {
  index <- match.arg(index)
  responses <- as.matrix(data[items])

  value <- switch(index,
    longstring = {
      apply(responses, 1, function(row) {
        runs <- rle(as.character(row[!is.na(row)]))
        if (length(runs$lengths) == 0) NA_real_ else max(runs$lengths)
      })
    },
    irv = {
      # Intra-individual response variability
      apply(responses, 1, stats::sd, na.rm = TRUE)
    },
    mahalanobis = {
      complete <- stats::complete.cases(responses)
      d2 <- rep(NA_real_, nrow(responses))
      d2[complete] <- stats::mahalanobis(responses[complete, , drop = FALSE],
        colMeans(responses[complete, , drop = FALSE]), stats::cov(responses[complete, ]))
      d2
    },
    even.odd = {
      if (is.null(scales)) {
        stop("`scales` is needed for `index = \"even.odd\"`.", call. = FALSE)
      }
      odd <- sapply(scales, function(s) rowMeans(data[s[c(TRUE, FALSE)]], na.rm = TRUE))
      even <- sapply(scales, function(s) rowMeans(data[s[c(FALSE, TRUE)]], na.rm = TRUE))
      vapply(seq_len(nrow(responses)), function(i) {
        stats::cor(odd[i, ], even[i, ], use = "complete.obs")
      }, numeric(1))
    },
    span = {
      apply(responses, 1, function(row) diff(range(row, na.rm = TRUE)))
    }
  )
  value <- unname(value)
  attr(value, "index") <- index
  value
}

#' Check attention items
#'
#' @description
#' Compares answers on instructed, infrequency and bogus items with the
#' expected answers.
#'
#' @param data A data frame of item responses, one row per respondent.
#' @param instructed,infrequency,bogus Named vectors giving the expected answer
#'   (value) for each attention item (name) of that kind.
#' @return A data frame with a failure count per kind and `n.failed`.
#' @export
attention.check <- function(data, instructed = NULL, infrequency = NULL, bogus = NULL) {
  out <- data.frame(row.names = seq_len(nrow(data)))

  # Kind `instructed`
  out$instructed <- 0L
  for (item in names(instructed)) {
    if (!item %in% names(data)) {
      stop(sprintf("Attention item `%s` (%s) is not a column of `data`.", item, "instructed"),
        call. = FALSE)
    }
    failed <- is.na(data[[item]]) | data[[item]] != instructed[[item]]
    out$instructed <- out$instructed + failed
  }

  # Kind `infrequency`
  out$infrequency <- 0L
  for (item in names(infrequency)) {
    if (!item %in% names(data)) {
      stop(sprintf("Attention item `%s` (%s) is not a column of `data`.", item, "infrequency"),
        call. = FALSE)
    }
    failed <- is.na(data[[item]]) | data[[item]] != infrequency[[item]]
    out$infrequency <- out$infrequency + failed
  }

  # Kind `bogus`
  out$bogus <- 0L
  for (item in names(bogus)) {
    if (!item %in% names(data)) {
      stop(sprintf("Attention item `%s` (%s) is not a column of `data`.", item, "bogus"),
        call. = FALSE)
    }
    failed <- is.na(data[[item]]) | data[[item]] != bogus[[item]]
    out$bogus <- out$bogus + failed
  }

  out$n.failed <- out$instructed + out$infrequency + out$bogus
  out
}

#' Flag speeders by completion time
#'
#' @description
#' Flags respondents who finished implausibly fast under one rule.
#'
#' @param seconds A numeric vector of completion times in seconds.
#' @param n.items Number of items in the questionnaire.
#' @param rule One of `"absolute"`, `"relative"`, `"per.item"` or `"mad"`.
#' @param limit The cutoff for the chosen rule: seconds for `"absolute"`, a
#'   share of the median for `"relative"`, seconds per item for
#'   `"per.item"` and a number of MADs for `"mad"`.
#' @return A logical vector the length of `seconds`.
#' @export
speed.flags <- function(seconds, n.items, rule = c("absolute", "relative", "per.item", "mad"),
                        limit = NULL) {
  rule <- match.arg(rule)
  defaults <- c(absolute = 60, relative = 0.5, per.item = 2, mad = 2)
  if (is.null(limit)) {
    limit <- defaults[[rule]]
  }
  typical <- stats::median(seconds, na.rm = TRUE)

  flag <- switch(rule,
    absolute = {
      seconds < limit
    },
    relative = {
      seconds < limit * typical
    },
    per.item = {
      seconds / n.items < limit
    },
    mad = {
      spread <- stats::mad(log(seconds), na.rm = TRUE)
      log(seconds) < log(typical) - limit * spread
    }
  )
  !is.na(flag) & flag
}

#' Summarise screening flags
#'
#' @description
#' Counts flagged respondents per check and overall.
#'
#' @param flags A data frame of logical flag columns as returned by
#'   [screen.responses()].
#' @param checks Character vector naming the flag columns to summarise.
#' @return A list with elements `n`, `per.check`, `share.per.check`,
#'   `n.any`, `share.any` and `keep`.
#' @export
screen.summary <- function(flags, checks = c("out.of.range", "longstring", "missing",
                                             "speeder")) {
  if (!all(checks %in% names(flags))) {
    stop("All `checks` must be columns of `flags`.", call. = FALSE)
  }
  hits <- as.matrix(flags[checks])

  # 1. Respondents
  n <- nrow(hits)
  # 2. Flags per check
  per.check <- colSums(hits)
  # 3. Share flagged per check
  share.per.check <- per.check / n
  # 4. Respondents with any flag
  n.any <- sum(rowSums(hits) > 0)
  # 5. Share with any flag
  share.any <- n.any / n
  # 6. Rows to keep
  keep <- unname(which(rowSums(hits) == 0))

  list(n = n, per.check = per.check, share.per.check = share.per.check,
    n.any = n.any, share.any = share.any, keep = keep)
}
