library(shiny)
library(bslib)

readings.path <- Sys.getenv("GREENHOUSE_LOG", "data/readings.csv")

thresholds <- list(
  temp_c = c(low = 12, high = 32),
  humidity = c(low = 45, high = 85),
  soil_moisture = c(low = 25, high = 70)
)

read.readings <- function(path) {
  raw <- read.csv(path, stringsAsFactors = FALSE)
  raw$time <- as.POSIXct(raw$time, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
  # the logger writes a partial line while it is flushing; drop it
  raw <- raw[!is.na(raw$time), , drop = FALSE]
  raw[order(raw$zone, raw$time), , drop = FALSE]
}

alert.level <- function(value, limits) {
  if (is.null(value) || is.na(value)) {
    return("unknown")
  }
  if (value < limits[["low"]]) {
    return("low")
  }
  if (value > limits[["high"]]) {
    return("high")
  }
  "ok"
}

latest.by.zone <- function(readings) {
  zones <- unique(readings$zone)
  if (length(zones) == 0) {
    return(readings[0, , drop = FALSE])
  }

  out <- vector("list", length(zones))
  for (i in seq_along(zones)) {
    rows <- readings[readings$zone == zones[i], , drop = FALSE]
    out[[i]] <- rows[which.max(rows$time), , drop = FALSE]
  }
  do.call(rbind, out)
}

draw.readings <- function(readings, measure, hours) {
  cutoff <- max(readings$time) - hours * 3600
  recent <- readings[readings$time >= cutoff, , drop = FALSE]
  limits <- thresholds[[measure]]
  zones <- unique(recent$zone)
  colours <- setNames(hcl.colors(length(zones), "Dark 3"), zones)

  # empty frame first, then the target band, then one line per zone on top
  plot(recent$time, recent[[measure]],
    type = "n", xlab = NULL, ylab = measure,
    ylim = range(c(recent[[measure]], limits), na.rm = TRUE)
  )
  rect(par("usr")[1], limits[["low"]], par("usr")[2], limits[["high"]],
    col = adjustcolor("palegreen", 0.25), border = NA
  )
  for (z in zones) {
    rows <- recent$zone == z
    lines(recent$time[rows], recent[[measure]][rows], col = colours[[z]], lwd = 2)
  }

  legend("topleft", legend = zones, col = colours, lwd = 2, bty = "n")
}

zone.card.ui <- function(id) {
  ns <- NS(id)
  card(
    card_header(textOutput(ns("title"), inline = TRUE)),
    uiOutput(ns("badges")),
    tableOutput(ns("latest"))
  )
}

zone.card.server <- function(id, readings, zone) {
  moduleServer(id, function(input, output, session) {
    latest <- reactive({
      rows <- readings()[readings()$zone == zone, , drop = FALSE]
      req(nrow(rows) > 0)
      rows[which.max(rows$time), , drop = FALSE]
    })

    output$title <- renderText(paste("Zone", zone))

    output$badges <- renderUI({
      levels <- vapply(names(thresholds), function(m) {
        alert.level(latest()[[m]], thresholds[[m]])
      }, character(1))
      bad <- levels[levels != "ok"]
      if (length(bad) == 0) {
        return(span(class = "badge bg-success", "All readings in range"))
      }
      tagList(lapply(names(bad), function(m) {
        span(class = "badge bg-danger me-1", paste(m, bad[[m]]))
      }))
    })

    output$latest <- renderTable({
      data.frame(
        Measure = names(thresholds),
        Value = vapply(names(thresholds), function(m) latest()[[m]], numeric(1)),
        At = format(latest()$time, "%H:%M")
      )
    })
  })
}

ui <- page_sidebar(
  title = "Greenhouse monitor",
  theme = bs_theme(version = 5, bootswatch = "minty"),
  sidebar = sidebar(
    selectInput("measure", "Measure", choices = names(thresholds)),
    sliderInput("hours", "Hours shown", min = 1, max = 72, value = 24),
    textOutput("last_update")
  ),
  card(card_header("Trend"), plotOutput("trend", height = "320px")),
  layout_column_wrap(width = 1 / 3, uiOutput("zone_cards"))
)

server <- function(input, output, session) {
  # re-read the log whenever the file changes, checking every 30 seconds
  readings <- reactiveFileReader(30000, session, readings.path, read.readings)
  zones <- reactive(sort(unique(readings()$zone)))

  output$zone_cards <- renderUI({
    lapply(zones(), function(z) zone.card.ui(paste0("zone_", z)))
  })

  # start each zone's module once; calling moduleServer again for the same
  # id would stack a second set of observers on top of the first
  started <- character()
  observeEvent(zones(), {
    for (z in setdiff(zones(), started)) {
      zone.card.server(paste0("zone_", z), readings, z)
    }
    started <<- union(started, zones())
  })

  output$trend <- renderPlot({
    validate(need(nrow(readings()) > 0, "No readings logged yet."))
    draw.readings(readings(), input$measure, input$hours)
  })

  output$last_update <- renderText({
    latest <- latest.by.zone(readings())
    stale <- latest$zone[latest$time < Sys.time() - 3600]
    if (length(stale) == 0) {
      return("All zones reported in the last hour.")
    }
    paste("No recent data from:", toString(stale))
  })
}

shinyApp(ui, server)
