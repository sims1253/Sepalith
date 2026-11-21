#' Check whether an action may be applied
#'
#' @param status A character vector of current interview statuses.
#' @param action One of `"approve"`, `"reject"`, `"flag"`, `"reassign"` or
#'   `"archive"`.
#' @return A logical vector the same length as `status`.
#' @keywords internal
#' @family action helpers
actionAllowed = function(status, action = c("approve", "reject", "flag", "reassign",
                                            "archive")) {
  action = match.arg(action)
  allowedFrom = list(
    approve = c("pending", "flagged"),
    reject = c("pending", "flagged"),
    flag = c("pending", "approved"),
    reassign = c("pending", "flagged", "rejected"),
    archive = c("approved", "rejected")
  )
  return(status %in% allowedFrom[[action]])
}

#' Apply a review action to interview records
#'
#' Rows whose status does not allow the action are left unchanged.
#'
#' @param records A data frame with columns `id`, `status`, `interviewer`
#'   and `flags`.
#' @param ids Identifiers of the records to change.
#' @param action One of `"approve"`, `"reject"`, `"flag"`, `"reassign"` or
#'   `"archive"`.
#' @param interviewer New interviewer, required when `action` is
#'   `"reassign"`.
#' @return `records` with the selected rows updated.
#' @export
#' @family action helpers
applyAction = function(records, ids, action = c("approve", "reject", "flag", "reassign",
                                                "archive"),
                       interviewer = NULL) {
  action = match.arg(action)
  rows = which(records$id %in% ids & actionAllowed(records$status, action))
  if (length(rows) == 0L) {
    return(records)
  }
  switch(action,
    approve = {
      records$status[rows] = "approved"
      records$flags[rows] = 0L
    },
    reject = {
      records$status[rows] = "rejected"
      records$flags[rows] = 0L
    },
    flag = {
      records$status[rows] = "flagged"
      records$flags[rows] = records$flags[rows] + 1L
    },
    reassign = {
      if (is.null(interviewer) || !nzchar(interviewer)) {
        stop("`interviewer` is required to reassign.", call. = FALSE)
      }
      records$status[rows] = "pending"
      records$interviewer[rows] = interviewer
    },
    archive = {
      records$status[rows] = "archived"
    }
  )
  return(records)
}

#' Explain why an action request cannot be applied
#'
#' @param records A data frame with columns `id`, `status`, `interviewer`
#'   and `flags`.
#' @param ids Identifiers of the records to change.
#' @param action Name of the requested action.
#' @param interviewer New interviewer, required when `action` is
#'   `"reassign"`.
#' @return `NULL` when the request can be applied, otherwise a single
#'   message for a notification.
#' @export
#' @family action helpers
actionRequestError = function(records, ids, action, interviewer = NULL) {
  # 1. ids: at least one, all known
  if (length(ids) == 0L) {
    return("Select at least one record.")
  }
  unknown = setdiff(ids, records$id)
  if (length(unknown) > 0L) {
    return(sprintf("Unknown record(s): %s.", paste(unknown, collapse = ", ")))
  }

  # 2. action: one of the known actions
  actions = c("approve", "reject", "flag", "reassign", "archive")
  if (length(action) != 1L || !action %in% actions) {
    return(sprintf("Unknown action: %s.", paste(action, collapse = ", ")))
  }

  # 3. interviewer: required for reassign
  if (action == "reassign" && (is.null(interviewer) || !nzchar(interviewer))) {
    return("Choose an interviewer to reassign to.")
  }

  # 4. status: every record must allow the action
  selected = records$id %in% ids
  blocked = records$id[selected & !actionAllowed(records$status, action)]
  if (length(blocked) > 0L) {
    return(sprintf("Cannot %s record(s): %s.", action, paste(blocked, collapse = ", ")))
  }
  return(NULL)
}

#' Create an action log entry
#'
#' @param action Name of the action that was applied.
#' @param ids Identifiers of the records it was applied to.
#' @param user Name of the user who applied it.
#' @param time Time stamp of the action.
#' @return A one-row data frame with columns `time`, `user`, `action`, `n`
#'   and `ids`.
#' @keywords internal
#' @family action helpers
actionLogEntry = function(action, ids, user, time = Sys.time()) {
  return(data.frame(
    time = time,
    user = user,
    action = action,
    n = length(ids),
    ids = paste(ids, collapse = ","),
    stringsAsFactors = FALSE
  ))
}

#' Count logged actions by type
#'
#' @param log A data frame of entries created by [actionLogEntry()].
#' @param user Only count actions by this user; `NULL` counts everyone.
#' @return A named integer vector with elements `approve`, `reject`, `flag`,
#'   `reassign` and `archive`: the number of records each action touched.
#' @export
#' @family action helpers
actionCounts = function(log, user = NULL) {
  actions = c("approve", "reject", "flag", "reassign", "archive")
  counts = stats::setNames(integer(length(actions)), actions)
  if (is.null(log) || nrow(log) == 0L) {
    return(counts)
  }
  if (!is.null(user)) {
    log = log[log$user == user, , drop = FALSE]
  }
  for (action in actions) {
    touched = log$n[log$action == action]
    counts[[action]] = as.integer(sum(touched))
  }
  return(counts)
}

#' Review actions UI
#'
#' @param id Module id.
#' @param interviewers Names offered when reassigning interviews.
#' @return A `shiny::tagList()` with one button per action.
#' @export
#' @family action module
actionsUi = function(id, interviewers) {
  ns = shiny::NS(id)
  return(shiny::tagList(
    shiny::actionButton(ns("approve"), "Approve", icon = shiny::icon("check"),
                        class = "btn-success"),
    shiny::actionButton(ns("reject"), "Reject", icon = shiny::icon("xmark"),
                        class = "btn-danger"),
    shiny::actionButton(ns("flag"), "Flag", icon = shiny::icon("flag"),
                        class = "btn-warning"),
    shiny::selectInput(ns("interviewer"), "Reassign to", choices = interviewers),
    shiny::actionButton(ns("reassign"), "Reassign", icon = shiny::icon("user"),
                        class = "btn-default"),
    shiny::actionButton(ns("archive"), "Archive", icon = shiny::icon("box-archive"),
                        class = "btn-default"),
    shiny::tableOutput(ns("counts"))
  ))
}

#' Review actions server
#'
#' @param id Module id.
#' @param store A `shiny::reactiveValues()` object with elements `records`
#'   and `log`.
#' @param selected A reactive returning the identifiers of the selected
#'   records.
#' @param user Name of the signed-in user.
#' @return `NULL`, invisibly; called for its side effects on `store`.
#' @export
#' @family action module
actionsServer = function(id, store, selected, user) {
  shiny::moduleServer(id, function(input, output, session) {
    # Approve the selected records
    shiny::observeEvent(input$approve, {
      ids = selected()
      shiny::req(length(ids) > 0L)
      store$records = applyAction(store$records, ids, "approve")
      store$log = rbind(store$log, actionLogEntry("approve", ids, user))
      shiny::showNotification(sprintf("%s: %d record(s).", "approve", length(ids)))
    })

    # Reject the selected records
    shiny::observeEvent(input$reject, {
      ids = selected()
      shiny::req(length(ids) > 0L)
      store$records = applyAction(store$records, ids, "reject")
      store$log = rbind(store$log, actionLogEntry("reject", ids, user))
      shiny::showNotification(sprintf("%s: %d record(s).", "reject", length(ids)))
    })

    # Flag the selected records
    shiny::observeEvent(input$flag, {
      ids = selected()
      shiny::req(length(ids) > 0L)
      store$records = applyAction(store$records, ids, "flag")
      store$log = rbind(store$log, actionLogEntry("flag", ids, user))
      shiny::showNotification(sprintf("%s: %d record(s).", "flag", length(ids)))
    })

    # Reassign the selected records
    shiny::observeEvent(input$reassign, {
      ids = selected()
      shiny::req(length(ids) > 0L, input$interviewer)
      store$records = applyAction(store$records, ids, "reassign", input$interviewer)
      store$log = rbind(store$log, actionLogEntry("reassign", ids, user))
      shiny::showNotification(sprintf("%s: %d record(s).", "reassign", length(ids)))
    })

    # Archive the selected records
    shiny::observeEvent(input$archive, {
      ids = selected()
      shiny::req(length(ids) > 0L)
      store$records = applyAction(store$records, ids, "archive")
      store$log = rbind(store$log, actionLogEntry("archive", ids, user))
      shiny::showNotification(sprintf("%s: %d record(s).", "archive", length(ids)))
    })

    output$counts = shiny::renderTable({
      counts = actionCounts(store$log, user)
      data.frame(action = names(counts), records = unname(counts))
    })
    return(invisible(NULL))
  })
}
