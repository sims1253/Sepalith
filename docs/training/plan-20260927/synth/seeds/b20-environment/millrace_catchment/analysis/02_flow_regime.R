# Flow regime of the Millrace outlet: duration curves, baseflow and flood
# frequency for the design report.

library(ggplot2)
library(gaugewise)

gauge <- read.csv("data/clean/gauge_daily.csv", stringsAsFactors = FALSE)
gauge$date <- as.Date(gauge$date)
gauge$wy <- water_year(gauge$date, start_month = 10L)

fig_dir <- "output/figures"
tab_dir <- "output/tables"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tab_dir, showWarnings = FALSE, recursive = TRUE)

complete_years <- function(df, min_days = 330) {
  counts <- tapply(!is.na(df$flow), df$wy, sum)
  keep <- as.integer(names(counts)[counts >= min_days])
  dropped <- setdiff(unique(df$wy), keep)
  if (length(dropped) > 0) {
    message("Dropping incomplete water years: ", paste(sort(dropped), collapse = ", "))
  }
  df[df$wy %in% keep, , drop = FALSE]
}

baseflow_by_year <- function(df, methods = c("one_param", "two_param", "fixed_interval")) {
  years <- sort(unique(df$wy))
  rows <- vector("list", length(years) * length(methods))
  k <- 0L
  for (y in years) {
    flow <- df$flow[df$wy == y]
    if (anyNA(flow)) {
      flow <- approx(seq_along(flow), flow, seq_along(flow), rule = 2)$y
    }
    for (m in methods) {
      k <- k + 1L
      sep <- separate_baseflow(flow, method = m)
      rows[[k]] <- data.frame(
        wy = y,
        method = m,
        bfi = baseflow_index(sep$flow, sep$baseflow),
        mean_base = mean(sep$baseflow)
      )
    }
  }
  do.call(rbind, rows[seq_len(k)])
}

frequency_table <- function(peaks, periods = c(2, 5, 10, 25, 50, 100)) {
  fits <- list(
    gumbel = fit_gumbel(peaks),
    lp3 = fit_lp3(peaks),
    gev = fit_gev(peaks)
  )
  tabs <- lapply(names(fits), function(nm) {
    tab <- design_flow(fits[[nm]], return_period = periods)
    tab$dist <- nm
    tab
  })
  out <- do.call(rbind, tabs)
  out[order(out$return_period, out$dist), c("dist", "return_period", "discharge")]
}

empirical_return_periods <- function(peaks) {
  n <- length(peaks)
  ord <- order(peaks, decreasing = TRUE)
  data.frame(
    peak_flow = peaks[ord],
    rank = seq_len(n),
    return_period = (n + 1) / seq_len(n)
  )
}

plot_fdc <- function(df) {
  curves <- lapply(split(df$flow, df$wy), fdc_exceedance)
  curves <- Map(function(cv, y) transform(cv, wy = as.integer(y)), curves, names(curves))
  curves <- do.call(rbind, curves)
  ggplot(curves, aes(exceedance * 100, flow, group = wy, colour = wy)) +
    geom_line(linewidth = 0.4, alpha = 0.7) +
    scale_y_log10() +
    scale_colour_viridis_c(name = "Water year") +
    labs(x = "Exceedance (%)", y = "Discharge (m3/s)", title = "Flow duration curves") +
    theme_minimal(base_size = 11)
}

plot_frequency <- function(design, empirical) {
  ggplot() +
    geom_line(
      data = design,
      aes(return_period, discharge, colour = dist),
      linewidth = 0.7
    ) +
    geom_point(
      data = empirical,
      aes(return_period, peak_flow),
      shape = 21, fill = "white"
    ) +
    scale_x_log10(breaks = c(1, 2, 5, 10, 25, 50, 100)) +
    labs(x = "Return period (years)", y = "Peak discharge (m3/s)", colour = NULL) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom")
}

annual_signatures <- function(df, low_quantile = 0.1, high_multiple = 3) {
  years <- sort(unique(df$wy))
  out <- data.frame(
    wy = years,
    mean_flow = NA_real_,
    q95 = NA_real_,
    high_pulses = NA_integer_,
    low_days = NA_integer_,
    centroid_day = NA_real_,
    flashiness = NA_real_
  )
  low_cut <- quantile(df$flow, low_quantile, na.rm = TRUE, names = FALSE)
  high_cut <- high_multiple * median(df$flow, na.rm = TRUE)
  for (i in seq_along(years)) {
    sub <- df[df$wy == years[i], , drop = FALSE]
    flow <- sub$flow
    ok <- !is.na(flow)
    if (sum(ok) < 300) {
      next
    }
    above <- flow > high_cut
    above[is.na(above)] <- FALSE
    pulses <- rle(above)
    day <- as.numeric(sub$date - min(sub$date)) + 1
    out$mean_flow[i] <- mean(flow, na.rm = TRUE)
    out$q95[i] <- fdc_q95(flow[ok])
    out$high_pulses[i] <- sum(pulses$values)
    out$low_days[i] <- sum(flow < low_cut, na.rm = TRUE)
    out$centroid_day[i] <- sum(day[ok] * flow[ok]) / sum(flow[ok])
    out$flashiness[i] <- sum(abs(diff(flow[ok]))) / sum(flow[ok])
  }
  out
}

save_figure <- function(plot, name, width = 7, height = 4.5) {
  path <- file.path(fig_dir, paste0(name, ".png"))
  ggsave(path, plot, width = width, height = height, dpi = 200)
  invisible(path)
}

# Flow duration ---------------------------------------------------------------

gauge_ok <- complete_years(gauge)
fdc_tab <- fdc_summary(gauge_ok, flow_col = "flow", group_col = "wy", min_days = 330)
write.csv(fdc_tab, file.path(tab_dir, "fdc_by_year.csv"), row.names = FALSE)
signatures <- annual_signatures(gauge_ok)
write.csv(signatures, file.path(tab_dir, "flow_signatures.csv"), row.names = FALSE)
save_figure(plot_fdc(gauge_ok), "fdc_by_year")

# Baseflow -------------------------------------------------------------------

bfi <- baseflow_by_year(gauge_ok)
write.csv(bfi, file.path(tab_dir, "bfi_by_year.csv"), row.names = FALSE)
print(aggregate(bfi ~ method, data = bfi, FUN = median))

rec <- recession_constant(na.omit(gauge_ok$flow))
cat(sprintf("Recession constant k = %.3f (half-life %.1f d)\n", rec$k, rec$half_life))

# Flood frequency -----------------------------------------------------------

peaks <- annual_maxima(gauge$date, gauge$flow, start_month = 10L, max_missing = 0.1)
design <- frequency_table(peaks$peak_flow)
write.csv(design, file.path(tab_dir, "design_floods.csv"), row.names = FALSE)

save_figure(
  plot_frequency(design, empirical_return_periods(peaks$peak_flow)),
  "flood_frequency"
)
