dsa_icer <- function(
  formula,
  data,
  param,
  range,
  ref,
  metric = "ICER",
  k = 1000
) {
  metric <- toupper(metric)
  if (metric == "NMB") {
    metric <- "INMB"
  }
  if (!metric %in% c("ICER", "INMB")) {
    stop("`metric` must be either 'ICER' or 'INMB'.", call. = FALSE)
  }
  if (!is.numeric(range)) {
    stop("`range` must be numeric.", call. = FALSE)
  }

  ce_data <- parse_ce_formula(formula, data, ref = ref, require_group = TRUE)
  source_names <- c(
    attr(ce_data, "cost_name"),
    attr(ce_data, "effect_name")
  )
  internal_param <- match(param, source_names)
  if (is.na(internal_param)) {
    stop(
      "`param` must be the cost or effect variable from `formula`.",
      call. = FALSE
    )
  }
  internal_param <- c("cost", "effect")[internal_param]

  values <- vapply(
    range,
    function(v) {
      modified <- ce_data
      modified[modified$group != ref, internal_param] <- v
      deltas <- ce_deltas(modified, ref)
      if (metric == "ICER") {
        deltas["ICER"]
      } else {
        k * deltas["delta_effect"] - deltas["delta_cost"]
      }
    },
    numeric(1)
  )

  results <- data.frame(Parameter = range, Value = values)
  names(results)[2] <- metric
  attr(results, "metric") <- metric
  results
}