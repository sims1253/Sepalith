#' Summarise a quota table
#'
#' @param quota A data frame with columns `cell`, `target`, `achieved` and
#'   `weight`.
#' @param slack Share of a cell's target by which it may differ and still
#'   be on track.
#' @return A list with elements `cells`, `target`, `achieved`, `pctDone`,
#'   `underCells` and `overCells`.
#' @export
#' @family tab helpers
summariseQuota = function(quota, slack = 0.05) {
  missing = setdiff(c("cell", "target", "achieved", "weight"), names(quota))
  if (length(missing) > 0L) {
    stop("Missing columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  # Totals
  cells = nrow(quota)
  target = sum(quota$target * quota$weight, na.rm = TRUE)
  achieved = sum(pmin(quota$achieved, quota$target) * quota$weight, na.rm = TRUE)
  pctDone = if (target > 0) achieved / target else NA_real_

  # Cells outside the slack band
  share = quota$achieved / quota$target
  underCells = quota$cell[!is.na(share) & share < 1 - slack]
  overCells = quota$cell[!is.na(share) & share > 1 + slack]
  return(list(
    cells = cells,
    target = target,
    achieved = achieved,
    pctDone = pctDone,
    underCells = underCells,
    overCells = overCells
  ))
}

#' Summarise interviewer performance
#'
#' @param responses A data frame of contact attempts with columns
#'   `interviewer`, `outcome` (`"complete"`, `"refusal"` or `"noncontact"`),
#'   `date` and `duration` (seconds).
#' @return A data frame with columns `interviewer`, `completes`, `refusals`,
#'   `rate`, `lastContact` and `avgDuration`, one row per interviewer.
#' @export
#' @family tab helpers
summariseInterviewers = function(responses) {
  who = factor(responses$interviewer)
  isComplete = responses$outcome == "complete"
  isRefusal = responses$outcome == "refusal"

  # 1. completes
  completes = tapply(isComplete, who, sum)

  # 2. refusals
  refusals = tapply(isRefusal, who, sum)

  # 3. rate: completes per attempt
  attempts = tapply(responses$outcome, who, length)
  rate = completes / attempts

  # 4. lastContact
  lastContact = tapply(as.numeric(responses$date), who, max)

  # 5. avgDuration: mean length of completed interviews
  avgDuration = tapply(ifelse(isComplete, responses$duration, NA), who, mean, na.rm = TRUE)
  avgDuration[is.nan(avgDuration)] = NA
  return(data.frame(
    interviewer = levels(who),
    completes = as.integer(completes),
    refusals = as.integer(refusals),
    rate = as.numeric(rate),
    lastContact = as.Date(as.numeric(lastContact), origin = "1970-01-01"),
    avgDuration = as.numeric(avgDuration),
    stringsAsFactors = FALSE
  ))
}

#' Count completed interviews over time
#'
#' @param responses A data frame of contact attempts with columns `outcome`
#'   and `date`.
#' @param by Time unit: one of `"day"`, `"week"` or `"month"`.
#' @return A data frame with columns `period` (first day of each period) and
#'   `completes`, sorted by `period`.
#' @export
#' @family tab helpers
timelineCounts = function(responses, by = c("day", "week", "month")) {
  by = match.arg(by)
  day = as.Date(responses$date[responses$outcome == "complete"])
  if (length(day) == 0L) {
    return(data.frame(period = as.Date(character(0)), completes = integer(0)))
  }
  period = switch(by,
    day = {
      day
    },
    week = {
      offset = as.integer(format(day, "%u")) - 1L
      day - offset
    },
    month = {
      offset = as.integer(format(day, "%d")) - 1L
      day - offset
    }
  )
  counts = table(period)
  return(data.frame(
    period = as.Date(names(counts)),
    completes = as.integer(counts),
    stringsAsFactors = FALSE
  ))
}

#' Filter contact attempts for the monitor tabs
#'
#' `NULL` or empty arguments leave the corresponding column unfiltered.
#'
#' @param responses A data frame of contact attempts with columns
#'   `interviewer`, `outcome`, `date` and `duration`.
#' @param interviewer Interviewers to keep.
#' @param outcome Outcomes to keep.
#' @param from First day to keep.
#' @param to Last day to keep.
#' @return The matching rows of `responses`.
#' @export
#' @family tab helpers
tabFilterResponses = function(responses, interviewer = NULL, outcome = NULL,
                              from = NULL, to = NULL) {
  keep = rep(TRUE, nrow(responses))
  day = as.Date(responses$date)

  # Keep the selected `interviewer`
  if (length(interviewer) > 0L) {
    keep = keep & responses$interviewer %in% interviewer
  }

  # Keep the selected `outcome`
  if (length(outcome) > 0L) {
    keep = keep & responses$outcome %in% outcome
  }

  # Keep days on or after `from`
  if (length(from) == 1L && !is.na(from)) {
    keep = keep & !is.na(day) & day >= as.Date(from)
  }

  # Keep days on or before `to`
  if (length(to) == 1L && !is.na(to)) {
    keep = keep & !is.na(day) & day <= as.Date(to)
  }
  return(responses[keep, , drop = FALSE])
}

#' Build tab titles with count badges
#'
#' @param counts A named list of counts with elements `overview`, `quotas`,
#'   `interviewers` and `timeline`; missing or zero counts show no badge.
#' @return A named character vector of tab titles.
#' @export
#' @family tab helpers
tabTitles = function(counts) {
  titles = c(overview = "Overview", quotas = "Quotas", interviewers = "Interviewers",
             timeline = "Timeline")
  for (tab in names(titles)) {
    n = counts[[tab]]
    if (!is.null(n) && !is.na(n) && n > 0) {
      titles[[tab]] = sprintf("%s (%d)", titles[[tab]], as.integer(n))
    }
  }
  return(titles)
}

#' Fieldwork monitor tabs UI
#'
#' @param id Module id.
#' @return A `shiny::tabsetPanel()` with one panel per section.
#' @export
#' @family tab module
tabsUi = function(id) {
  ns = shiny::NS(id)
  titles = tabTitles(list())
  return(shiny::tabsetPanel(
    id = ns("tabs"),
    shiny::tabPanel(
      titles[["overview"]], value = "overview",
      shiny::verbatimTextOutput(ns("overview"))
    ),
    shiny::tabPanel(
      titles[["quotas"]], value = "quotas",
      shiny::tableOutput(ns("quotas"))
    ),
    shiny::tabPanel(
      titles[["interviewers"]], value = "interviewers",
      shiny::tableOutput(ns("interviewers"))
    ),
    shiny::tabPanel(
      titles[["timeline"]], value = "timeline",
      shiny::radioButtons(ns("by"), NULL, c("day", "week", "month"), inline = TRUE),
      shiny::plotOutput(ns("timeline"))
    )
  ))
}

#' Fieldwork monitor tabs server
#'
#' @param id Module id.
#' @param responses A reactive returning the contact attempts.
#' @param quota A reactive returning the quota table.
#' @param wave A reactive returning the wave summary list used by
#'   [formatWaveSummary()].
#' @return `NULL`, invisibly; called for its outputs.
#' @export
#' @family tab module
tabsServer = function(id, responses, quota, wave) {
  shiny::moduleServer(id, function(input, output, session) {
    # Overview tab
    output$overview = shiny::renderText({
      shiny::req(wave())
      paste(formatWaveSummary(wave()), collapse = "\n")
    })

    # Quotas tab
    output$quotas = shiny::renderTable({
      problems = checkQuotaTable(quota())
      shiny::validate(shiny::need(length(problems) == 0L, paste(problems, collapse = "\n")))
      q = quota()
      q$progress = vapply(seq_len(nrow(q)), function(i) {
        formatProgress(q$achieved[[i]], q$target[[i]])$label
      }, character(1))
      q
    })

    # Interviewers tab
    output$interviewers = shiny::renderTable({
      shiny::validate(shiny::need(nrow(responses()) > 0L, "No contact attempts yet."))
      formatInterviewerTable(summariseInterviewers(responses()))$table
    })

    # Timeline tab
    output$timeline = shiny::renderPlot({
      counts = timelineCounts(responses(), input$by)
      shiny::validate(shiny::need(nrow(counts) > 0L, "No completed interviews yet."))
      graphics::barplot(counts$completes, names.arg = format(counts$period), las = 2)
    })
    return(invisible(NULL))
  })
}
