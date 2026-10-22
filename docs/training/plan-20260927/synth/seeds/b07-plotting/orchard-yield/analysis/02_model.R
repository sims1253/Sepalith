library(dplyr)
library(tidyr)
library(ggplot2)
library(plotsmith)
library(tabulora)

harvest <- readRDS("data/derived/harvest_clean.rds")
model_dir <- "output/models"
fig_dir <- "output/figures"
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

summarise_plot_season <- function(df) {
  df %>%
    filter(!outlier, !is.na(yield_per_tree)) %>%
    group_by(plot_id, block, rootstock, variety, season) %>%
    summarise(
      yield_per_tree = sum(yield_kg, na.rm = TRUE) / max(trees_picked, na.rm = TRUE),
      n_picks = n(),
      first_pick = min(harvest_date),
      .groups = "drop"
    ) %>%
    mutate(
      log_yield = log(yield_per_tree),
      season_c = season - round(mean(season))
    )
}

fit_base_model <- function(df) {
  lm(log_yield ~ variety + rootstock + season_c, data = df)
}

fit_block_model <- function(df) {
  lm(log_yield ~ variety + rootstock + season_c + block, data = df)
}

fit_interaction_model <- function(df) {
  lm(log_yield ~ variety * rootstock + season_c + block, data = df)
}

compare_models <- function(models) {
  tibble(
    model = names(models),
    df = vapply(models, function(m) as.integer(attr(logLik(m), "df")), integer(1)),
    aic = vapply(models, AIC, numeric(1)),
    bic = vapply(models, BIC, numeric(1)),
    r2_adj = vapply(models, function(m) summary(m)$adj.r.squared, numeric(1))
  ) %>%
    mutate(delta_aic = aic - min(aic)) %>%
    arrange(aic)
}

predict_variety_means <- function(model, df) {
  grid <- df %>%
    distinct(variety, rootstock) %>%
    mutate(season_c = 0, block = levels(df$block)[1])
  pred <- predict(model, newdata = grid, interval = "confidence")
  grid %>%
    bind_cols(as_tibble(pred)) %>%
    mutate(across(c(fit, lwr, upr), exp)) %>%
    rename(estimate = fit, lower = lwr, upper = upr)
}

plot_variety_means <- function(means) {
  p <- ggplot(means, aes(x = variety, y = estimate, colour = rootstock)) +
    geom_pointrange(
      aes(ymin = lower, ymax = upper),
      position = position_dodge(width = 0.5)
    ) +
    scale_colour_plotsmith() +
    theme_plotsmith(grid = "y")
  label_plot(
    p,
    x = NULL,
    y = label_with_unit("Predicted yield per tree", "kg"),
    title = "Predicted yield by variety and rootstock",
    subtitle = "Back-transformed means at the reference season, 95% CI"
  )
}

plot_season <- summarise_plot_season(harvest)

models <- list(
  base = fit_base_model(plot_season),
  block = fit_block_model(plot_season),
  interaction = fit_interaction_model(plot_season)
)

model_table <- compare_models(models)
print(model_table)

best_name <- model_table$model[1]
best <- models[[best_name]]
message("Selected model: ", best_name)

saveDiagnostics(best, file.path(fig_dir, "diagnostics_best_model.png"),
                title = paste("Diagnostics:", best_name, "model"))

means <- predict_variety_means(best, plot_season)
p_means <- plot_variety_means(means)
save_plot(p_means, "variety_means", dir = fig_dir, formats = "png")

saveRDS(plot_season, file.path(model_dir, "plot_season.rds"))
saveRDS(models, file.path(model_dir, "models.rds"))
write.csv(model_table, file.path(model_dir, "model_comparison.csv"), row.names = FALSE)
write.csv(means, file.path(model_dir, "variety_means.csv"), row.names = FALSE)
