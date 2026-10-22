options(modelcrate.registry_dir = file.path(tempdir(), "mc-registry-check"))

# registry_dir
stopifnot(dir.exists(registry_dir()))

# register_model / list_versions
toy <- structure(list(k = 2), class = c("mc_toy", "mc_model"))
stopifnot(identical(register_model(toy, "churn.v", metrics = c(rmse = 1.5)), "1"))
stopifnot(identical(register_model(toy, "churn.v"), "2"))
register_model(toy, "churnXv", version = "7")
stopifnot(identical(list_versions("churn.v"), c("1", "2")))
stopifnot(inherits(try(register_model(toy, "churn.v", "2"), silent = TRUE), "try-error"))

# remove_model, then the next default version must not reuse "2"
stopifnot(isTRUE(remove_model("churn.v", "1")), !remove_model("churn.v", "99"))
stopifnot(identical(register_model(toy, "churn.v"), "3"))

# load_model reads from disk after the memory copy is gone
rm(list = "churn.v@3", envir = .registry)
stopifnot(identical(load_model("churn.v")$version, "3"))
stopifnot(inherits(try(load_model("nope"), silent = TRUE), "try-error"))

# registry_table
tab <- registry_table("rmse")
stopifnot(nrow(tab) == 3, all(tab$class == "mc_toy"))

# predict_registered
predict.mc_toy <- function(object, newdata, ...) rep(object$k, nrow(newdata))
out <- predict_registered("churn.v", data.frame(a = 1:3))
stopifnot(identical(out$.pred, c(2, 2, 2)), all(out$version == "3"), all(out$model == "churn.v"))
