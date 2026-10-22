# skipped: needs xgboost and Matrix (xgb_dmatrix, fit_xgb_model, predict.mc_xgb,
# xgb_importance_table, xgb_eval_history, print.mc_xgb)

# xgb_encode_outcome
reg <- xgb_encode_outcome(c(1.5, 2))
stopifnot(reg$objective == "reg:squarederror", is.null(reg$classes))
bin <- xgb_encode_outcome(c("no", "yes", "no"))
stopifnot(bin$objective == "binary:logistic", identical(bin$label, c(0L, 1L, 0L)))
multi <- xgb_encode_outcome(factor(c("b", "a", "c")))
stopifnot(multi$eval_metric == "mlogloss", identical(multi$classes, c("a", "b", "c")))
stopifnot(identical(multi$label, c(1L, 0L, 2L)))
