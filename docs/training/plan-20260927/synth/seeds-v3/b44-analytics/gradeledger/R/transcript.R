#' Validate course records
#'
#' Checks the columns `student`, `term`, `course`, `credits` and `points` in turn.
#'
#' @param records A data frame with columns `student`, `term`, `course`,
#'   `credits` and `points` (grade points, 0 to 4), one row per course taken.
#' @export
#' @return `records`, invisibly, when they are valid.
validateRecords = function(records) {
  required = c("student", "term", "course", "credits", "points")
  missing = setdiff(required, names(records))
  if (length(missing) > 0L) {
    stop(sprintf("`records` is missing column(s): %s.", toString(missing)), call. = FALSE)
  }

  # Check `student`
  bad = is.na(records$student) | records$student == ""
  if (any(bad)) {
    stop(sprintf("Column `%s` is invalid in row(s) %s.", "student", toString(which(bad))),
      call. = FALSE)
  }
  # Check `term`
  bad = is.na(records$term) | records$term == ""
  if (any(bad)) {
    stop(sprintf("Column `%s` is invalid in row(s) %s.", "term", toString(which(bad))),
      call. = FALSE)
  }
  # Check `course`
  bad = is.na(records$course) | records$course == ""
  if (any(bad)) {
    stop(sprintf("Column `%s` is invalid in row(s) %s.", "course", toString(which(bad))),
      call. = FALSE)
  }
  # Check `credits`
  bad = is.na(records$credits) | records$credits <= 0
  if (any(bad)) {
    stop(sprintf("Column `%s` is invalid in row(s) %s.", "credits", toString(which(bad))),
      call. = FALSE)
  }
  # Check `points`
  bad = is.na(records$points) | records$points < 0 | records$points > 4
  if (any(bad)) {
    stop(sprintf("Column `%s` is invalid in row(s) %s.", "points", toString(which(bad))),
      call. = FALSE)
  }

  return(invisible(records))
}

#' Summarise each term
#'
#' Courses with zero grade points count as attempted but not earned.
#'
#' @inheritParams validateRecords
#' @export
#' @return A data frame with columns `term`, `courses`, `attempted`, `earned`,
#'   `failed`, `qualityPoints` and `gpa`, one row per term.
termSummary = function(records) {
  terms = sort(unique(records$term))
  f = factor(records$term, levels = terms)
  passed = records$points > 0
  out = data.frame(term = terms)

  # Field `courses`
  value = tapply(records$course, f, length, default = 0L)
  out$courses = as.integer(value)
  # Field `attempted`
  value = tapply(records$credits, f, sum, default = 0)
  out$attempted = as.numeric(value)
  # Field `earned`
  value = tapply(records$credits * passed, f, sum, default = 0)
  out$earned = as.numeric(value)
  # Field `failed`
  value = tapply(!passed, f, sum, default = 0L)
  out$failed = as.integer(value)
  # Field `qualityPoints`
  value = tapply(records$credits * records$points, f, sum, default = 0)
  out$qualityPoints = as.numeric(value)
  # Field `gpa`
  value = out$qualityPoints / out$attempted
  out$gpa = round(value, 2)

  return(out)
}

#' Running grade point average
#'
#' @inheritParams validateRecords
#' @export
#' @return A data frame with columns `term`, `termGpa`, `credits` (cumulative
#'   attempted) and `cumGpa`, terms in sorted order.
cumulativeGpa = function(records) {
  terms = sort(unique(records$term))
  n = length(terms)
  out = data.frame(term = terms, termGpa = NA_real_, credits = NA_real_, cumGpa = NA_real_)
  attempted = 0
  quality = 0
  for (i in seq_len(n)) {
    rows = records$term == terms[i]
    termCredits = sum(records$credits[rows])
    termQuality = sum(records$credits[rows] * records$points[rows])
    attempted = attempted + termCredits
    quality = quality + termQuality
    out$termGpa[i] = round(termQuality / termCredits, 2)
    out$credits[i] = attempted
    out$cumGpa[i] = round(quality / attempted, 2)
  }
  return(out)
}

#' Academic standing flags
#'
#' Applies one rule per flag: `deansList`, `honors`, `warning`, `probation`
#' and `underload`.
#'
#' @param cumGpa Cumulative grade point averages.
#' @param termGpa Grade point averages for the current term.
#' @param credits Credits attempted in the current term.
#' @param thresholds Named limits `deansList`, `honors`, `warning`, `probation`
#'   and `minCredits`.
#' @export
#' @return A data frame with one logical column per flag, `count` and `standing`.
academicStanding = function(cumGpa, termGpa, credits,
                            thresholds = c(deansList = 3.7, honors = 3.5, warning = 2,
                                           probation = 1.7, minCredits = 12)) {
  n = length(cumGpa)
  flags = data.frame(row.names = seq_len(n))
  count = integer(n)

  # Rule `deansList`
  limit = thresholds[["deansList"]]
  hit = !is.na(termGpa) & termGpa >= limit & credits >= thresholds[["minCredits"]]
  flags$deansList = hit
  count = count + hit
  # Rule `honors`
  limit = thresholds[["honors"]]
  hit = !is.na(cumGpa) & cumGpa >= limit
  flags$honors = hit
  count = count + hit
  # Rule `warning`
  limit = thresholds[["warning"]]
  hit = !is.na(termGpa) & termGpa < limit
  flags$warning = hit
  count = count + hit
  # Rule `probation`
  limit = thresholds[["probation"]]
  hit = !is.na(cumGpa) & cumGpa < limit
  flags$probation = hit
  count = count + hit
  # Rule `underload`
  limit = thresholds[["minCredits"]]
  hit = !is.na(credits) & credits < limit
  flags$underload = hit
  count = count + hit

  flags$count = count
  flags$standing = ifelse(flags$probation, "probation",
                          ifelse(flags$warning, "warning", "good"))
  rownames(flags) = NULL
  return(flags)
}

#' Convert percentages to a national grading scale
#'
#' @param percent A numeric vector of percentages.
#' @param system One of `"ects"`, `"uk"`, `"german"` or `"ib"`.
#' @export
#' @return A data frame with columns `percent`, `grade` and `passed`.
convertScale = function(percent, system = c("ects", "uk", "german", "ib")) {
  system = match.arg(system)
  if (!is.numeric(percent)) {
    stop("`percent` must be numeric.", call. = FALSE)
  }
  switch(system,
    ects = {
      cuts = c(0, 50, 60, 70, 80, 90)
      labels = c("F", "E", "D", "C", "B", "A")
      passMark = 50
    },
    uk = {
      cuts = c(0, 40, 50, 60, 70)
      labels = c("Fail", "Third", "2:2", "2:1", "First")
      passMark = 40
    },
    german = {
      cuts = c(0, 50, 60, 70, 80, 90)
      labels = c("5.0", "4.0", "3.0", "2.0", "1.5", "1.0")
      passMark = 50
    },
    ib = {
      cuts = c(0, 15, 30, 45, 60, 75, 85)
      labels = c("1", "2", "3", "4", "5", "6", "7")
      passMark = 45
    }
  )

  idx = findInterval(percent, cuts)
  idx[idx == 0L] = NA_integer_
  return(data.frame(percent = percent, grade = labels[idx], passed = percent >= passMark))
}

#' Format a transcript as text lines
#'
#' Shows `term`, `course`, `credits` and `points` for each record.
#'
#' @inheritParams validateRecords
#' @param format One of `"text"`, `"csv"`, `"tsv"` or `"markdown"`.
#' @export
#' @return A character vector with a header line followed by one line per record.
transcriptLines = function(records, format = c("text", "csv", "tsv", "markdown")) {
  format = match.arg(format)
  cols = c("term", "course", "credits", "points")
  records = records[order(records$term, records$course), cols]
  switch(format,
    text = {
      header = sprintf("%-8s %-10s %7s %6s", "term", "course", "credits", "points")
      rows = sprintf("%-8s %-10s %7.1f %6.2f", records$term, records$course,
                     records$credits, records$points)
      lines = c(header, rows)
    },
    csv = {
      header = paste(cols, collapse = ",")
      rows = sprintf("%s,%s,%g,%g", records$term, records$course,
                     records$credits, records$points)
      lines = c(header, rows)
    },
    tsv = {
      header = paste(cols, collapse = "\t")
      rows = sprintf("%s\t%s\t%g\t%g", records$term, records$course,
                     records$credits, records$points)
      lines = c(header, rows)
    },
    markdown = {
      header = c(sprintf("| %s |", paste(cols, collapse = " | ")), "|---|---|---:|---:|")
      rows = sprintf("| %s | %s | %g | %g |", records$term, records$course,
                     records$credits, records$points)
      lines = c(header, rows)
    }
  )
  return(lines)
}
