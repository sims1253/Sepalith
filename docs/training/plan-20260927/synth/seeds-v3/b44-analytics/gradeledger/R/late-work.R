#' Deduct a late penalty
#'
#' @param score Scores before the penalty, in percent.
#' @param hoursLate Hours past the deadline; zero, negative or `NA` means on time.
#' @param policy One of `"none"`, `"flat"`, `"perDay"`, `"perHour"`, `"tiered"`
#'   or `"cutoff"`.
#' @param rate Deduction in percentage points per unit of lateness.
#' @param cutoffHours Hours after which `"cutoff"` deducts everything.
#' @return A data frame with columns `score`, `hoursLate`, `deduction` and `final`.
#' @export
latePenalty = function(score, hoursLate,
                       policy = c("none", "flat", "perDay", "perHour", "tiered", "cutoff"),
                       rate = 10, cutoffHours = 72) {
  policy = match.arg(policy)
  hours = pmax(ifelse(is.na(hoursLate), 0, hoursLate), 0)
  late = hours > 0
  days = ceiling(hours / 24)
  switch(policy,
    none = {
      # No deduction
      deduction = rep(0, length(score))
    },
    flat = {
      # `rate` once the work is late
      deduction = ifelse(late, rate, 0)
    },
    perDay = {
      # `rate` per started day
      deduction = rate * days
    },
    perHour = {
      # `rate` per started day, spread over its hours
      deduction = rate / 24 * ceiling(hours)
    },
    tiered = {
      # `rate` for the first day, twice `rate` per later day
      deduction = ifelse(days <= 1, rate * days, rate + 2 * rate * (days - 1))
    },
    cutoff = {
      # `rate` per started day, everything after `cutoffHours`
      deduction = rate * days
      deduction[hours > cutoffHours] = 100
    }
  )

  deduction = pmin(deduction, 100)
  final = score * (1 - deduction / 100)
  return(data.frame(score = score, hoursLate = hours, deduction = deduction,
                    final = round(final, 2)))
}

#' Review extension requests
#'
#' Applies the rules `lateRequest`, `tooLong`, `tooMany` and `undocumented`;
#' a request is approved when no rule fires.
#'
#' @param requests A data frame with columns `student`, `requested` and `due`
#'   (date-times), `days` and `documented`, in the order received.
#' @param maxDays Longest extension granted, in days.
#' @param maxPerStudent Most extensions granted to one student.
#' @param noticeHours Fewest hours between request and deadline.
#' @param docDays Longest extension granted without documentation, in days.
#' @return A data frame with columns `student`, one logical column per rule,
#'   `approved`, `reasons` and `newDue`.
#' @export
reviewExtensions = function(requests, maxDays = 14, maxPerStudent = 2, noticeHours = 24,
                            docDays = 3) {
  n = nrow(requests)
  reasons = character(n)
  flags = data.frame(student = requests$student)

  # Rule `lateRequest`
  notice = as.numeric(difftime(requests$due, requests$requested, units = "hours"))
  hit = notice < noticeHours
  flags$lateRequest = hit
  reasons[hit] = paste0(reasons[hit], "lateRequest;")
  # Rule `tooLong`
  hit = requests$days > maxDays
  flags$tooLong = hit
  reasons[hit] = paste0(reasons[hit], "tooLong;")
  # Rule `tooMany`
  nth = stats::ave(seq_len(n), requests$student, FUN = seq_along)
  hit = nth > maxPerStudent
  flags$tooMany = hit
  reasons[hit] = paste0(reasons[hit], "tooMany;")
  # Rule `undocumented`
  hit = requests$days > docDays & !requests$documented
  flags$undocumented = hit
  reasons[hit] = paste0(reasons[hit], "undocumented;")

  flags$approved = reasons == ""
  flags$reasons = sub(";$", "", reasons)
  newDue = requests$due
  newDue[flags$approved] = newDue[flags$approved] + requests$days[flags$approved] * 86400
  flags$newDue = newDue
  return(flags)
}

#' Classify submissions by timeliness
#'
#' Statuses are assigned in the order `excused`, `missing`, `onTime`, `grace`
#' and `late`; the first that applies wins.
#'
#' @param due,submitted Deadlines and submission times as date-times;
#'   `submitted` is `NA` when nothing was handed in.
#' @param excused Logical, whether the work was excused.
#' @param graceMinutes Minutes after the deadline still counted as `grace`.
#' @return A list with elements `status`, `minutesLate` and `counts`.
#' @export
classifySubmissions = function(due, submitted, excused = FALSE, graceMinutes = 15) {
  excused = rep_len(excused, length(due))
  minutesLate = as.numeric(difftime(submitted, due, units = "mins"))
  status = rep(NA_character_, length(due))
  counts = c(excused = 0L, missing = 0L, onTime = 0L, grace = 0L, late = 0L)

  # Status `excused`
  hit = is.na(status) & excused
  status[hit] = "excused"
  counts[["excused"]] = sum(hit)
  # Status `missing`
  hit = is.na(status) & is.na(submitted)
  status[hit] = "missing"
  counts[["missing"]] = sum(hit)
  # Status `onTime`
  hit = is.na(status) & minutesLate <= 0
  status[hit] = "onTime"
  counts[["onTime"]] = sum(hit)
  # Status `grace`
  hit = is.na(status) & minutesLate <= graceMinutes
  status[hit] = "grace"
  counts[["grace"]] = sum(hit)
  # Status `late`
  hit = is.na(status)
  status[hit] = "late"
  counts[["late"]] = sum(hit)

  return(list(status = status, minutesLate = pmax(minutesLate, 0), counts = counts))
}

#' Timeliness per student
#'
#' @param student Student ids, one per submission.
#' @param status Statuses as returned by [classifySubmissions()].
#' @return A data frame with columns `student`, `onTime`, `grace`, `late`,
#'   `missing`, `excused`, `total` and `lateShare`.
#' @export
lateSummary = function(student, status) {
  students = sort(unique(student))
  s = factor(student, levels = students)
  out = data.frame(student = students)

  # Count `onTime`
  hit = status == "onTime"
  out$onTime = as.integer(tapply(hit, s, sum, default = 0L))
  # Count `grace`
  hit = status == "grace"
  out$grace = as.integer(tapply(hit, s, sum, default = 0L))
  # Count `late`
  hit = status == "late"
  out$late = as.integer(tapply(hit, s, sum, default = 0L))
  # Count `missing`
  hit = status == "missing"
  out$missing = as.integer(tapply(hit, s, sum, default = 0L))
  # Count `excused`
  hit = status == "excused"
  out$excused = as.integer(tapply(hit, s, sum, default = 0L))

  out$total = as.integer(table(s))
  counted = pmax(out$total - out$excused, 1L)
  out$lateShare = round((out$late + out$missing) / counted, 3)
  return(out)
}

#' Compare late policies side by side
#'
#' @param hours Hours late to preview.
#' @param score Score before the penalty, in percent.
#' @param rate Passed to [latePenalty()].
#' @return A matrix of final scores with one row per value of `hours` and one
#'   column per policy.
#' @export
penaltyPreview = function(hours = c(1, 12, 24, 48, 96), score = 100, rate = 10) {
  policies = c("none", "flat", "perDay", "perHour", "tiered", "cutoff")
  out = matrix(NA_real_, nrow = length(hours), ncol = length(policies),
               dimnames = list(paste0(hours, "h"), policies))
  for (p in policies) {
    res = latePenalty(rep(score, length(hours)), hours, policy = p, rate = rate)
    out[, p] = res$final
  }
  return(out)
}
