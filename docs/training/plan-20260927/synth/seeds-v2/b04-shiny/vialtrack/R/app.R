#' Launch the sample tracker
#'
#' Starts the vialtrack Shiny application against the sample registry stored
#' at `db_path`.
#'
#' @param db_path Path to the SQLite registry file.
#' @param lab_name Display name of the lab shown in the navbar.
#' @param ... Passed to [shiny::shinyApp()] via `options`.
#' @return A `shiny.appobj`, which runs when printed.
#' @importFrom shiny shinyApp
#' @export
run_app <- function(db_path = default_db_path(), lab_name = "Sample Lab", ...) {
  if (!file.exists(db_path)) {
    stop("Registry not found at ", db_path, call. = FALSE)
  }

  app <- shinyApp(ui = app_ui, server = app_server, options = list(...))
  # read back inside the app with get_app_option()
  app$appOptions$vialtrack <- list(db_path = db_path, lab_name = lab_name)
  return(app)
}

#' Application user interface
#'
#' Builds the top-level navbar page with one panel per module.
#'
#' @param request Internal parameter for bookmarking.
#' @return A [bslib::page_navbar()] object.
#' @importFrom bslib page_navbar nav_panel bs_theme
#' @keywords internal
app_ui <- function(request) {
  page <- page_navbar(
    title = get_app_option("lab_name", "Sample Lab"),
    theme = bs_theme(version = 5, preset = "flatly"),
    header = add_external_resources(),
    nav_panel("Register", mod_sample_entry_ui("entry")),
    nav_panel("Freezers", mod_freezer_map_ui("freezers")),
    nav_panel("About", app_about_panel())
  )
  return(page)
}

#' Application server logic
#'
#' Opens the registry connection, wires the modules together and closes the
#' connection when the session ends.
#'
#' @param input,output,session Internal parameters for Shiny.
#' @return Called for its side effects; returns `NULL` invisibly.
#' @importFrom shiny onStop reactiveVal
#' @keywords internal
app_server <- function(input, output, session) {
  con <- DBI::dbConnect(RSQLite::SQLite(), get_app_option("db_path", default_db_path()))
  onStop(function() DBI::dbDisconnect(con))
  registry <- reactiveVal(read_registry(con))

  new_sample <- mod_sample_entry_server("entry", registry = registry)
  mod_freezer_map_server("freezers", registry = registry)

  # persist accepted samples, then refresh so both modules see the new row
  shiny::observeEvent(new_sample(), {
    DBI::dbAppendTable(con, "samples", new_sample())
    registry(read_registry(con))
  })

  return(invisible(NULL))
}

#' Read an application option
#'
#' Looks up a value stored by [run_app()] in the running app's options,
#' falling back to `default` outside of a running app.
#'
#' @param name The name of the option.
#' @param default Value returned when the option is not set.
#' @return The option value, or `default`.
#' @importFrom shiny getShinyOption
#' @noRd
get_app_option <- function(name, default = NULL) {
  opts <- getShinyOption("vialtrack", default = list())
  if (is.null(opts[[name]])) {
    return(default)
  }
  return(opts[[name]])
}

#' Locate the default registry file
#'
#' Uses the `VIALTRACK_DB` environment variable when set, otherwise a file
#' in the user's data directory.
#'
#' @return A single file path.
#' @noRd
default_db_path <- function() {
  # an explicit path wins, e.g. for a shared registry on a network drive
  env_path <- Sys.getenv("VIALTRACK_DB", unset = "")
  if (nzchar(env_path)) {
    return(normalizePath(env_path, mustWork = FALSE))
  }
  return(file.path(tools::R_user_dir("vialtrack", "data"), "registry.sqlite"))
}

#' Read the sample registry
#'
#' Pulls all active samples and parses the collection timestamp.
#'
#' @param con A DBI connection to the registry.
#' @return A data frame of samples ordered by collection time.
#' @importFrom magrittr %>%
#' @keywords internal
read_registry <- function(con) {
  # discarded vials stay in the table for the audit trail but are hidden here
  samples <- DBI::dbGetQuery(con, "SELECT * FROM samples WHERE discarded = 0") %>%
    transform(collected_at = as.POSIXct(collected_at, tz = "UTC"))
  return(samples[order(samples$collected_at), , drop = FALSE])
}

#' Add external resources to the page head
#'
#' Registers the package's `www` directory and links the custom stylesheet
#' and favicon.
#'
#' @importFrom shiny addResourcePath tags
#' @keywords internal
add_external_resources <- function() {
  www <- system.file("app", "www", package = "vialtrack")
  addResourcePath("vialtrack-www", www)

  head <- tags$head(
    tags$link(rel = "stylesheet", href = "vialtrack-www/custom.css"),
    tags$link(rel = "icon", href = "vialtrack-www/favicon.ico")
  )
  return(head)
}

#' About panel content
#'
#' A short description of the app with the package version.
#'
#' @importFrom bslib card card_header card_body
#' @keywords internal
app_about_panel <- function() {
  version <- as.character(utils::packageVersion("vialtrack"))
  panel <- card(
    card_header("About vialtrack"),
    card_body(
      shiny::p("Register incoming samples and track where each vial is stored."),
      shiny::p(paste("Version", version))
    )
  )
  return(panel)
}
