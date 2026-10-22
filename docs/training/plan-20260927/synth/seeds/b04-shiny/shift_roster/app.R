library(shiny)
library(bslib)
library(dplyr)
library(DT)

shiftTypes = c(early = "06:00-14:00", late = "14:00-22:00", night = "22:00-06:00")

staffList = data.frame(
  name = c("Amara", "Bastian", "Chiara", "Dmitri", "Elif", "Farid"),
  maxShifts = c(5, 5, 4, 5, 3, 5),
  nightsOk = c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE),
  stringsAsFactors = FALSE
)

emptyRoster = function(weekStart) {
  days = seq(weekStart, by = "day", length.out = 7)
  expand.grid(date = days, shift = names(shiftTypes), stringsAsFactors = FALSE) %>%
    mutate(staff = NA_character_) %>%
    arrange(date, match(shift, names(shiftTypes)))
}

assignShift = function(roster, date, shift, staff) {
  row = which(roster$date == date & roster$shift == shift)
  if (length(row) != 1) {
    stop("No slot for ", format(date), " ", shift, call. = FALSE)
  }
  roster$staff[row] = staff
  roster
}

clearShift = function(roster, date, shift) {
  row = which(roster$date == date & roster$shift == shift)
  if (length(row) != 1) {
    stop("No slot for ", format(date), " ", shift, call. = FALSE)
  }
  roster$staff[row] = NA_character_
  roster
}

rosterProblems = function(roster, people) {
  problems = character()
  assigned = roster %>% filter(!is.na(staff))
  sameDay = assigned %>% count(date, staff) %>% filter(n > 1)
  if (nrow(sameDay) > 0) {
    problems = c(problems, sprintf("%s has two shifts on %s", sameDay$staff, sameDay$date))
  }
  load = assigned %>%
    count(staff, name = "shifts") %>%
    inner_join(people, by = c("staff" = "name")) %>%
    filter(shifts > maxShifts)
  if (nrow(load) > 0) {
    over = sprintf("%s is over their limit (%d/%d)", load$staff, load$shifts, load$maxShifts)
    problems = c(problems, over)
  }
  nights = assigned %>%
    filter(shift == "night") %>%
    inner_join(people, by = c("staff" = "name")) %>%
    filter(!nightsOk)
  if (nrow(nights) > 0) {
    problems = c(problems, sprintf("%s cannot work nights (%s)", nights$staff, nights$date))
  }
  gaps = roster %>% filter(is.na(staff))
  if (nrow(gaps) > 0) {
    problems = c(problems, sprintf("%d shifts are still unassigned", nrow(gaps)))
  }
  problems
}

rosterWide = function(roster) {
  roster %>%
    mutate(day = format(date, "%a %d %b")) %>%
    select(day, shift, staff) %>%
    tidyr::pivot_wider(names_from = shift, values_from = staff)
}

staffHours = function(roster) {
  roster %>%
    filter(!is.na(staff)) %>%
    count(staff, name = "shifts") %>%
    right_join(select(staffList, staff = name), by = "staff") %>%
    mutate(shifts = coalesce(shifts, 0L), hours = shifts * 8) %>%
    arrange(desc(hours), staff)
}

ui = page_navbar(
  title = "Shift roster",
  theme = bs_theme(version = 5, bootswatch = "cosmo"),
  nav_panel(
    "Plan",
    layout_sidebar(
      sidebar = sidebar(
        dateInput("weekStart", "Week starting", value = Sys.Date(), weekstart = 1),
        selectInput("date", "Day", choices = NULL),
        radioButtons("shift", "Shift", choices = names(shiftTypes), inline = TRUE),
        selectInput("staff", "Staff member", choices = staffList$name),
        actionButton("assign", "Assign", class = "btn-primary"),
        actionButton("clear", "Clear slot"),
        downloadButton("export", "Export CSV")
      ),
      card(card_header("Week"), DTOutput("rosterTable")),
      card(card_header("Problems"), uiOutput("problems"))
    )
  ),
  nav_panel("Hours", card(tableOutput("hours")))
)

server = function(input, output, session) {
  roster = reactiveVal(NULL)
  observeEvent(input$weekStart, {
    roster(emptyRoster(input$weekStart))
    days = seq(input$weekStart, by = "day", length.out = 7)
    updateSelectInput(session, "date", choices = setNames(format(days), format(days, "%A %d %b")))
  })
  observeEvent(input$assign, {
    req(input$date, input$staff)
    roster(assignShift(roster(), as.Date(input$date), input$shift, input$staff))
  })
  observeEvent(input$clear, {
    req(input$date)
    roster(clearShift(roster(), as.Date(input$date), input$shift))
  })
  output$rosterTable = renderDT({
    req(roster())
    datatable(rosterWide(roster()), rownames = FALSE, options = list(dom = "t", ordering = FALSE))
  })
  output$problems = renderUI({
    req(roster())
    found = rosterProblems(roster(), staffList)
    if (length(found) == 0) {
      return(p(class = "text-success", "No problems found."))
    }
    tags$ul(lapply(found, tags$li))
  })
  output$hours = renderTable(staffHours(req(roster())))
  output$export = downloadHandler(
    filename = function() paste0("roster-", format(input$weekStart), ".csv"),
    content = function(file) {
      out = roster() %>% mutate(time = unname(shiftTypes[shift]))
      write.csv(out, file, row.names = FALSE, na = "")
    }
  )
}

shinyApp(ui, server)
