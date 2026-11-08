#' Fill gaps in a student roster
#'
#' Replaces missing or empty values of `year`, `group`, `program` and
#' `support` with the matching element of `defaults`.
#'
#' @param students A data frame with column `student` and any of `year`,
#'   `group`, `program` and `support`.
#' @param defaults A named list of fill values for `year`, `group`, `program`
#'   and `support`.
#' @return `students` with all four columns complete, and an attribute
#'   `filled` counting the values filled per column.
#' @export
fill.roster <- function(students,
                        defaults = list(year = "unknown", group = "unassigned",
                                        program = "general", support = FALSE)) {
  filled <- c(year = 0L, group = 0L, program = 0L, support = 0L)

  # Fill `year`
  if (is.null(students$year)) students$year <- NA
  miss <- is.na(students$year) | students$year == ""
  students$year[miss] <- defaults$year
  filled[["year"]] <- sum(miss)
  # Fill `group`
  if (is.null(students$group)) students$group <- NA
  miss <- is.na(students$group) | students$group == ""
  students$group[miss] <- defaults$group
  filled[["group"]] <- sum(miss)
  # Fill `program`
  if (is.null(students$program)) students$program <- NA
  miss <- is.na(students$program) | students$program == ""
  students$program[miss] <- defaults$program
  filled[["program"]] <- sum(miss)
  # Fill `support`
  if (is.null(students$support)) students$support <- NA
  miss <- is.na(students$support) | students$support == ""
  students$support[miss] <- defaults$support
  filled[["support"]] <- sum(miss)

  attr(students, "filled") <- filled
  students
}

#' Attendance broken down by cohort
#'
#' One block per roster dimension `year`, `group`, `program` and `support`.
#'
#' @param summary A data frame with columns `student` and `attended.rate`, as
#'   returned by [student.summary()].
#' @inheritParams fill.roster
#' @param threshold Rate below which a student counts in `below`.
#' @param digits Number of decimal places kept in `mean.rate`.
#' @return A data frame with columns `dimension`, `level`, `students`,
#'   `mean.rate` and `below`.
#' @export
cohort.breakdown <- function(summary, students, threshold = 0.9, digits = 3) {
  data <- merge(summary[c("student", "attended.rate")], students, by = "student")
  out <- list()

  # Breakdown by `year`
  g <- factor(data$year)
  rate <- tapply(data$attended.rate, g, mean)
  low <- tapply(data$attended.rate < threshold, g, sum)
  out$year <- data.frame(dimension = "year", level = levels(g), students = as.vector(table(g)),
                         mean.rate = round(as.vector(rate), digits), below = as.vector(low))
  # Breakdown by `group`
  g <- factor(data$group)
  rate <- tapply(data$attended.rate, g, mean)
  low <- tapply(data$attended.rate < threshold, g, sum)
  out$group <- data.frame(dimension = "group", level = levels(g), students = as.vector(table(g)),
                          mean.rate = round(as.vector(rate), digits), below = as.vector(low))
  # Breakdown by `program`
  g <- factor(data$program)
  rate <- tapply(data$attended.rate, g, mean)
  low <- tapply(data$attended.rate < threshold, g, sum)
  out$program <- data.frame(dimension = "program", level = levels(g),
                            students = as.vector(table(g)),
                            mean.rate = round(as.vector(rate), digits), below = as.vector(low))
  # Breakdown by `support`
  g <- factor(data$support)
  rate <- tapply(data$attended.rate, g, mean)
  low <- tapply(data$attended.rate < threshold, g, sum)
  out$support <- data.frame(dimension = "support", level = levels(g),
                            students = as.vector(table(g)),
                            mean.rate = round(as.vector(rate), digits), below = as.vector(low))

  out <- do.call(rbind, out)
  rownames(out) <- NULL
  out
}

#' Plain text attendance report
#'
#' Writes the chosen sections in the order `overview`, `attendance`, `flags`
#' and `bands`, each under a `## <section>` heading.
#'
#' @param summary A data frame as returned by [student.summary()].
#' @param flags A data frame as returned by [attendance.flags()].
#' @param sections Any of `"overview"`, `"attendance"`, `"flags"` and `"bands"`.
#' @return A character vector of report lines.
#' @export
report.lines <- function(summary, flags,
                         sections = c("overview", "attendance", "flags", "bands")) {
  sections <- match.arg(sections, several.ok = TRUE)
  lines <- character(0)

  # Section `overview`
  if ("overview" %in% sections) {
    body <- sprintf("%d students, %d sessions", nrow(summary), sum(summary$sessions))
    lines <- c(lines, "## overview", body, "")
  }
  # Section `attendance`
  if ("attendance" %in% sections) {
    body <- sprintf("%s: %.1f%%", summary$student, 100 * summary$attended.rate)
    lines <- c(lines, "## attendance", body, "")
  }
  # Section `flags`
  if ("flags" %in% sections) {
    body <- sprintf("%s: %d flag(s)", flags$student, as.integer(flags$n.flags))
    lines <- c(lines, "## flags", body, "")
  }
  # Section `bands`
  if ("bands" %in% sections) {
    band <- cut(summary$attended.rate, c(-Inf, 0.9, 0.95, Inf), right = FALSE,
                labels = c("red", "amber", "green"))
    body <- sprintf("%s: %s", summary$student, band)
    lines <- c(lines, "## bands", body, "")
  }

  lines[-length(lines)]
}

#' Write report tables to disk
#'
#' @param tables A named list of data frames; names become file names.
#' @param dir Directory to write into; created when missing.
#' @param format One of `"csv"`, `"tsv"`, `"rds"` or `"txt"`; `import.report()`
#'   does not read `"txt"`.
#' @return The written paths, named by table, invisibly.
#' @rdname report-io
#' @export
export.report <- function(tables, dir, format = c("csv", "tsv", "rds", "txt")) {
  format <- match.arg(format)
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  paths <- stats::setNames(character(length(tables)), names(tables))
  for (name in names(tables)) {
    path <- file.path(dir, paste0(name, ".", format))
    switch(format,
      csv = {
        utils::write.csv(tables[[name]], path, row.names = FALSE)
      },
      tsv = {
        utils::write.table(tables[[name]], path, sep = "\t", quote = FALSE, row.names = FALSE)
      },
      rds = {
        saveRDS(tables[[name]], path)
      },
      txt = {
        writeLines(utils::capture.output(print(tables[[name]], row.names = FALSE)), path)
      }
    )
    paths[[name]] <- path
  }
  invisible(paths)
}

#' @return For `import.report()`, a named list of data frames, one per file.
#' @rdname report-io
#' @export
import.report <- function(dir, format = c("csv", "tsv", "rds")) {
  format <- match.arg(format)
  files <- sort(list.files(dir, pattern = paste0("\\.", format, "$"), full.names = TRUE))
  tables <- stats::setNames(vector("list", length(files)),
                            sub(paste0("\\.", format, "$"), "", basename(files)))
  for (i in seq_along(files)) {
    path <- files[i]
    switch(format,
      csv = {
        tables[[i]] <- utils::read.csv(path, stringsAsFactors = FALSE)
      },
      tsv = {
        tables[[i]] <- utils::read.delim(path, quote = "", stringsAsFactors = FALSE)
      },
      rds = {
        tables[[i]] <- readRDS(path)
      }
    )
  }
  tables
}

#' Weekly attendance per cohort
#'
#' @param records A data frame with columns `student`, `date`, `session` and
#'   `status`.
#' @inheritParams fill.roster
#' @param by One of `"year"`, `"group"` or `"program"`.
#' @return A tibble with columns named by `by`, `week`, `sessions` and `rate`.
#' @importFrom dplyr %>%
#' @importFrom rlang .data
#' @export
weekly.cohort <- function(records, students, by = c("year", "group", "program")) {
  by <- match.arg(by)
  key <- rlang::sym(by)
  records %>%
    dplyr::mutate(
      week = format(.data$date, "%G-W%V"),
      attended = .data$status %in% c("present", "late", "left")
    ) %>%
    dplyr::inner_join(students, by = "student") %>%
    dplyr::group_by(!!key, .data$week) %>%
    dplyr::summarise(sessions = dplyr::n(), rate = mean(.data$attended), .groups = "drop") %>%
    dplyr::arrange(!!key, .data$week)
}
