# skipped: run_app, app_ui, app_server, get_app_option, add_external_resources,
#   app_about_panel (need shiny / bslib); read_registry (needs DBI, magrittr)

old <- Sys.getenv("VIALTRACK_DB", unset = NA)
Sys.setenv(VIALTRACK_DB = "/tmp/vt/registry.sqlite")
stopifnot(identical(default_db_path(), normalizePath("/tmp/vt/registry.sqlite", mustWork = FALSE)))
Sys.unsetenv("VIALTRACK_DB")
stopifnot(basename(default_db_path()) == "registry.sqlite", grepl("vialtrack", default_db_path()))
if (!is.na(old)) Sys.setenv(VIALTRACK_DB = old)
