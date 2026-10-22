library(ggplot2)
library(plotsmith)
library(tabulora)

plot_season <- readRDS("output/models/plot_season.rds")
models <- readRDS("output/models/models.rds")
model_table <- read.csv("output/models/model_comparison.csv")
fig_dir <- "output/figures/report"
tab_dir <- "output/tables"
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

season_totals <- function(df) {
  agg <- aggregate(yield_per_tree ~ season + variety, data = df, FUN = mean)
  agg[order(agg$variety, agg$season), ]
}

season_change <- function(totals) {
  seasons <- sort(unique(totals$season))
  if (length(seasons) < 2) {
    stop("Need at least two seasons to compute a change")
  }

  last <- seasons[length(seasons)]
  prev <- seasons[length(seasons) - 1]
  wide <- merge(
    totals[totals$season == prev, c("variety", "yield_per_tree")],
    totals[totals$season == last, c("variety", "yield_per_tree")],
    by = "variety", suffixes = c("_prev", "_last")
  )

  wide$change <- format_percent_change(wide$yield_per_tree_prev, wide$yield_per_tree_last)
  wide
}

plot_trend_by_variety <- function(totals) {
  p <- ps_line(totals, x = "season", y = "yield_per_tree", group = "variety") +
    scale_colour_plotsmith() +
    scale_x_continuous(breaks = sort(unique(totals$season)))

  p |>
    label_plot(
      x = "Season",
      y = label_with_unit("Mean yield per tree", "kg"),
      title = "Yield trend by variety"
    ) |>
    add_caption_source("Orchard harvest log", date = Sys.Date())
}

plot_rootstock_panels <- function(df) {
  p <- ps_scatter(df, x = "first_pick", y = "yield_per_tree", colour = "variety", size = 1.5) |>
    add_trend_line(method = "loess", se = FALSE)

  # Rootstock groups differ a lot in vigour, so free y scales per panel
  p <- facet_wrap_balanced(p, "rootstock", max_cols = 3, scales = "free_y") +
    strip_theme_top() +
    scale_colour_plotsmith() +
    theme_legend_position("bottom")

  label_plot(p, x = "First pick date", y = "Yield per tree (kg)",
             title = "Earlier picks and yield, by rootstock")
}

plot_block_series_base <- function(df, path) {
  png(path, width = 1600, height = 1000, res = 160)
  on.exit(dev.off(), add = TRUE)

  agg <- aggregate(yield_per_tree ~ season + block, data = df, FUN = median)
  cols <- plot_series(agg$season, agg$yield_per_tree, agg$block, legend_pos = "bottomleft")
  title(main = "Median yield per tree by block", ylab = "kg")
  invisible(cols)
}

write_tables <- function(data, model_table, change, best_model, dir) {
  md_summary <- kable_summary(data, vars = c("yield_per_tree", "n_picks"),
                              caption = "Plot-season summary")
  md_models <- kable_report(model_table, digits = 1, caption = "Model comparison")
  md_coefs <- kable_coefficients(best_model, caption = "Coefficients of the selected model")
  md_change <- kable_report(change, caption = "Change between the last two seasons")
  writeLines(c(md_summary, "", md_models, "", md_coefs, "", md_change),
             file.path(dir, "report_tables.md"))

  # The HTML version flags varieties whose yield dropped
  html <- gt_report(change, title = "Season-on-season change",
                    subtitle = "Mean yield per tree in kg") |>
    gt_format_by_type(decimals = 2) |>
    gt_highlight_rows(grepl("^-", change$change), fill = "#fde2e1") |>
    gt_add_source("Orchard harvest log")
  gt::gtsave(html, file.path(dir, "season_change.html"))

  invisible(dir)
}

totals <- season_totals(plot_season)
change <- season_change(totals)
best_model <- models[[model_table$model[1]]]

report_plots <- list(
  yield_trend = plot_trend_by_variety(totals),
  rootstock_panels = plot_rootstock_panels(plot_season)
)
save_plot_set(report_plots, dir = fig_dir, formats = c("png", "pdf"), preset = "slide")
plot_block_series_base(plot_season, file.path(fig_dir, "block_series.png"))

write_tables(plot_season, model_table, change, best_model, tab_dir)
message("Report outputs written to ", fig_dir, " and ", tab_dir)
