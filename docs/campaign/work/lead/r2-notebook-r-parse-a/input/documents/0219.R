.getDialog <- function(name, newInstance = FALSE) {
  if (bindingIsLocked(".dialogs", asNamespace("Deducer"))) {
    unlockBinding(".dialogs", asNamespace("Deducer"))
  }
  dialog <- .dialogs[[name]]
  if (is.null(dialog) || newInstance) {
    dialog <- .dialogGenerators[[name]]()
    if (!newInstance) {
      di <- .dialogs
      di[[name]] <- dialog
      .dialogs <<- di
    }
  }
  dialog
}