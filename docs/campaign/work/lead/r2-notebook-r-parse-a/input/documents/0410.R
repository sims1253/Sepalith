plot.alarm_plot_data <- function(x, ...) {
  # Extract data
  data <- x$epidemic_data
  best_models <- x$best_models

  # Prepare alert data
  alert_data <- do.call(
    rbind,
    lapply(names(best_models), function(model_name) {
      model_data <- best_models[[model_name]]
      subset(model_data, model_data$Alarm == 1, select = c("Date", "ScYr"))
    })
  )
  alert_data$Model <- rep(
    names(best_models),
    sapply(best_models, function(x) sum(x$Alarm == 1))
  )

  alert_data <- unique(alert_data)

  # Assign consistent y-positions to each model
  unique_models <- unique(alert_data$Model)
  model_positions <- setNames(
    seq(-1, -length(unique_models), by = -1),
    unique_models
  )

  alert_data$y_position <- model_positions[alert_data$Model]

  # Combine epidemic data with alert data
  plot_data <- merge(data, alert_data, by = c("Date", "ScYr"), all.x = TRUE)
  plot_data <- unique(plot_data)

  # Create a plot for each year
  plots <- lapply(split(plot_data, plot_data$ScYr), function(year_data) {
    year <- unique(year_data$ScYr)

    # Calculate y-axis limits
    y_max <- max(
      c(year_data$lab_conf, year_data$pct_absent * 100),
      na.rm = TRUE
    )
    y_min <- min(model_positions) - 1

    p <- ggplot(year_data, aes(x = .data$Date)) +
      # Absenteeism percentage
      geom_area(
        aes(y = .data$pct_absent * 100, fill = "Absenteeism (%)"),
        alpha = 0.7,
        show.legend = FALSE
      ) +
      # Lab confirmed cases
      geom_area(
        aes(y = .data$lab_conf, fill = "Lab Confirmed Cases"),
        alpha = 0.7,
        show.legend = FALSE
      ) +
      # Reference date
      geom_vline(
        data = dplyr::filter(year_data, .data$ref_date == 1),
        aes(xintercept = .data$Date, color = "Reference Date"),
        linetype = "dashed"
      ) +
      # Alert points (stacked)
      geom_point(
        data = dplyr::filter(year_data, !is.na(.data$Model)),
        aes(y = .data$y_position, color = .data$Model),
        shape = 15,
        size = 3
      ) +
      scale_fill_manual(
        values = c(
          "Absenteeism (%)" = "grey70",
          "Lab Confirmed Cases" = "black"
        )
      ) +

      scale_color_manual(
        values = c(
          "Reference Date" = "orange",
          setNames(scales::hue_pal()(length(unique_models)), unique_models)
        )
      ) +
      scale_y_continuous(
        name = "Average Absenteeism Percentage",
        sec.axis = sec_axis(~., name = "Confirmed Influenza Cases"),
        limits = c(y_min, y_max * 1.1)
      ) + # Extend y-axis below 0 for stacked points

      labs(
        title = paste("Epidemic Data with Alerts - Year", year),
        x = "Time",
        fill = "Data Type",
        color = "Alerts and Reference"
      ) +

      theme_bw() +
      theme(
        legend.position = "bottom",
        axis.title.y.right = element_text(color = "black"),
        axis.title.y.left = element_text(color = "grey50"),
        legend.box.background = element_rect(),
        axis.title = element_text(size = 10),
        plot.title = element_text(size = 10)
      )

    return(p)
  })

  return(plots)
}