#' Tenure in years at a reference date
#'
#' Measures time from hire to the earlier of termination and `as_of`. Staff
#' hired after `as_of` get `NA`.
#'
#' @param hire_date Date vector of hire dates.
#' @param term_date Date vector of termination dates; `NA` for current staff.
#' @param as_of Reference date.
#' @return A numeric vector of tenure in years.
#' @export
tenure_years <- function(hire_date, term_date, as_of = Sys.Date()) {
  end <- pmin(dplyr::coalesce(term_date, as_of), as_of)
  years <- as.numeric(difftime(end, hire_date, units = "days")) / 365.25
  years[hire_date > as_of] <- NA_real_
  return(years)
}

#' Bucket tenure into reporting bands
#'
#' Cuts tenure in years into the bands used on the people dashboard.
#'
#' @param years Numeric vector of tenure in years.
#' @param breaks Upper bounds of each band in years, ascending.
#' @return An ordered factor of tenure bands.
#' @export
tenure_band <- function(years, breaks = c(1, 3, 5, 10)) {
  labels <- c(
    paste0("<", breaks[1], "y"),
    paste0(utils::head(breaks, -1), "-", breaks[-1], "y"),
    paste0(breaks[length(breaks)], "y+")
  )
  band <- cut(years, breaks = c(0, breaks, Inf), labels = labels, right = FALSE,
    ordered_result = TRUE)
  return(band)
}

#' Headcount on a given date
#'
#' Counts employees hired on or before `date` and not yet terminated.
#'
#' @param staff A data frame of employees with columns `employee_id`,
#'   `hire_date` and `term_date`.
#' @param date A single date.
#' @return A single integer.
#' @keywords internal
headcount_at <- function(staff, date) {
  active <- staff$hire_date <= date &
    (is.na(staff$term_date) | staff$term_date > date)
  return(sum(active))
}

#' Hires within a period
#'
#' Selects employees whose hire date falls in `[from, to]`.
#'
#' @param staff A data frame of employees with columns `employee_id`,
#'   `hire_date` and `term_date`.
#' @param from,to Dates bounding the period, inclusive.
#' @return A subset of `staff`.
#' @family period movements
#' @export
hires_in_period <- function(staff, from, to) {
  if (from > to) {
    stop("`from` must not be after `to`.", call. = FALSE)
  }
  out <- staff %>%
    dplyr::filter(hire_date >= from, hire_date <= to)
  return(out)
}

#' Leavers within a period
#'
#' Selects employees whose termination date falls in `[from, to]`.
#'
#' @param staff A data frame of employees with columns `employee_id`,
#'   `hire_date` and `term_date`.
#' @param from,to Dates bounding the period, inclusive.
#' @return A subset of `staff`.
#' @family period movements
#' @export
leavers_in_period <- function(staff, from, to) {
  if (from > to) {
    stop("`from` must not be after `to`.", call. = FALSE)
  }
  out <- staff %>%
    dplyr::filter(!is.na(term_date), term_date >= from, term_date <= to)
  return(out)
}

#' Monthly attrition rate
#'
#' For each month, divides leavers by the average of the opening and closing
#' headcount. Annualised turnover is twelve times the monthly mean.
#'
#' @param staff A data frame of employees with columns `employee_id`,
#'   `hire_date` and `term_date`.
#' @param from,to Dates bounding the period, inclusive.
#' @return A data frame with columns `month`, `opening`, `closing`, `leavers`
#'   and `rate`, with attribute `annualised`.
#' @export
monthly_attrition <- function(staff, from, to) {
  starts <- seq(lubridate::floor_date(from, "month"), to, by = "month")
  n <- length(starts)
  opening <- integer(n)
  closing <- integer(n)
  leavers <- integer(n)
  for (i in seq_len(n)) {
    month_end <- lubridate::ceiling_date(starts[i], "month") - 1
    opening[i] <- headcount_at(staff, starts[i] - 1)
    closing[i] <- headcount_at(staff, month_end)
    leavers[i] <- nrow(leavers_in_period(staff, starts[i], month_end))
  }
  avg_headcount <- (opening + closing) / 2
  rate <- ifelse(avg_headcount > 0, leavers / avg_headcount, NA_real_)
  out <- data.frame(
    month = format(starts, "%Y-%m"),
    opening = opening,
    closing = closing,
    leavers = leavers,
    rate = rate
  )
  attr(out, "annualised") <- 12 * mean(rate, na.rm = TRUE)
  return(out)
}

#' Retention curves by hire cohort
#'
#' Groups employees by hire quarter or year and computes, for each cohort,
#' the share still employed after each whole number of months.
#'
#' @param staff A data frame of employees with columns `employee_id`,
#'   `hire_date` and `term_date`.
#' @param cohort Granularity of the hire cohorts.
#' @param horizon Number of months to follow each cohort.
#' @param as_of Reference date; months after it are left as `NA`.
#' @return A data frame with columns `cohort`, `size`, `month` and
#'   `retained`.
#' @export
cohort_retention <- function(staff, cohort = c("quarter", "year"), horizon = 24L,
                             as_of = Sys.Date()) {
  cohort <- match.arg(cohort)
  staff$cohort <- switch(cohort,
    quarter = paste0(format(staff$hire_date, "%Y"), "-Q",
      (as.integer(format(staff$hire_date, "%m")) - 1) %/% 3 + 1),
    year = format(staff$hire_date, "%Y")
  )
  cohorts <- sort(unique(staff$cohort))
  months <- seq_len(horizon)
  retained <- matrix(NA_real_, nrow = length(cohorts), ncol = horizon)
  sizes <- integer(length(cohorts))
  for (i in seq_along(cohorts)) {
    members <- staff[staff$cohort == cohorts[i], , drop = FALSE]
    sizes[i] <- nrow(members)
    last_hire <- max(members$hire_date)
    for (m in months) {
      step <- lubridate::period(m, "months")
      checkpoint <- lubridate::add_with_rollback(members$hire_date, step)
      if (lubridate::add_with_rollback(last_hire, step) > as_of) {
        break
      }
      still_here <- is.na(members$term_date) | members$term_date > checkpoint
      retained[i, m] <- mean(still_here)
    }
  }
  out <- data.frame(
    cohort = rep(cohorts, times = horizon),
    size = rep(sizes, times = horizon),
    month = rep(months, each = length(cohorts)),
    retained = as.vector(retained)
  )
  return(out[order(out$cohort, out$month), ])
}

#' Attrition by employee group
#'
#' Compares leavers and average headcount across a grouping column such as
#' department or tenure band over one period.
#'
#' @param staff A data frame of employees with columns `employee_id`,
#'   `hire_date` and `term_date`.
#' @param group Name of the grouping column.
#' @param from,to Dates bounding the period, inclusive.
#' @param min_size Groups with a smaller average headcount are suppressed.
#' @return A data frame with columns for the group, `headcount`, `leavers`,
#'   `suppressed` and `rate`.
#' @export
attrition_by_group <- function(staff, group, from, to, min_size = 5L) {
  if (!group %in% names(staff)) {
    stop("Column `", group, "` not found in `staff`.", call. = FALSE)
  }
  pieces <- split(staff, staff[[group]])
  out <- lapply(names(pieces), function(g) {
    part <- pieces[[g]]
    headcount <- (headcount_at(part, from) + headcount_at(part, to)) / 2
    leavers <- nrow(leavers_in_period(part, from, to))
    data.frame(group = g, headcount = headcount, leavers = leavers)
  })
  out <- do.call(rbind, out)
  out$suppressed <- out$headcount < min_size
  out$rate <- ifelse(out$suppressed | out$headcount == 0, NA_real_,
    out$leavers / out$headcount)
  names(out)[1] <- group
  return(out[order(-out$rate, na.last = TRUE), ])
}
