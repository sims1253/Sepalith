library(shiny)
library(bslib)
library(dplyr)
library(tidyr)
library(ggplot2)

grade_breaks <- c(0, 50, 60, 70, 80, 90, 100)
grade_labels <- c("F", "E", "D", "C", "B", "A")

check_response_columns <- function(data) {
  required <- c("student_id", "question", "answer")
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) {
    return(paste("Responses file is missing:", toString(missing)))
  }
  if (anyDuplicated(data[c("student_id", "question")]) > 0) {
    return("Responses file has more than one answer per student and question.")
  }
  NULL
}

check_key_columns <- function(data) {
  required <- c("question", "correct", "points")
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) {
    return(paste("Answer key is missing:", toString(missing)))
  }
  if (anyDuplicated(data$question) > 0) {
    return("Answer key lists the same question more than once.")
  }
  NULL
}

read_upload <- function(file) {
  readr::read_csv(file$datapath, col_types = readr::cols(.default = "c")) |>
    mutate(across(everything(), trimws))
}

score_responses <- function(responses, key, case_sensitive = FALSE) {
  key <- key |> mutate(points = as.numeric(points))
  scored <- key |>
    left_join(responses, by = "question", relationship = "one-to-many") |>
    mutate(
      given = if (case_sensitive) answer else tolower(answer),
      expected = if (case_sensitive) correct else tolower(correct),
      earned = if_else(!is.na(given) & given == expected, points, 0)
    )
  scored |>
    filter(!is.na(student_id)) |>
    group_by(student_id) |>
    summarise(
      answered = sum(!is.na(answer)),
      score = sum(earned),
      .groups = "drop"
    ) |>
    mutate(
      percent = round(100 * score / sum(key$points), 1),
      grade = cut(percent, grade_breaks, grade_labels, include.lowest = TRUE, right = FALSE)
    ) |>
    arrange(desc(percent), student_id)
}

item_difficulty <- function(responses, key) {
  responses |>
    inner_join(key, by = "question") |>
    group_by(question) |>
    summarise(
      attempts = n(),
      correct_rate = mean(tolower(answer) == tolower(correct)),
      .groups = "drop"
    ) |>
    mutate(flag = case_when(
      correct_rate < 0.3 ~ "very hard",
      correct_rate > 0.95 ~ "very easy",
      TRUE ~ ""
    )) |>
    arrange(correct_rate)
}

plot_score_distribution <- function(scores, pass_mark) {
  ggplot(scores, aes(x = percent)) +
    geom_histogram(binwidth = 5, boundary = 0, fill = "#6a51a3", colour = "white") +
    geom_vline(xintercept = pass_mark, linetype = "dashed", colour = "#cb181d") +
    scale_x_continuous(limits = c(0, 100)) +
    labs(x = "Score (%)", y = "Students") +
    theme_minimal(base_size = 13)
}

ui <- page_sidebar(
  title = "Quiz scorer",
  sidebar = sidebar(
    fileInput("responses", "Responses (CSV)", accept = ".csv"),
    fileInput("key", "Answer key (CSV)", accept = ".csv"),
    checkboxInput("case_sensitive", "Case-sensitive answers", value = FALSE),
    sliderInput("pass_mark", "Pass mark (%)", min = 0, max = 100, value = 50, step = 5),
    downloadButton("download_scores", "Download scores")
  ),
  navset_card_tab(
    nav_panel("Scores", DT::DTOutput("scores")),
    nav_panel("Distribution", plotOutput("distribution")),
    nav_panel("Questions", tableOutput("difficulty"))
  )
)

server <- function(input, output, session) {
  responses <- reactive({
    req(input$responses)
    data <- read_upload(input$responses)
    validate(check_response_columns(data))
    data
  })
  key <- reactive({
    req(input$key)
    data <- read_upload(input$key)
    validate(check_key_columns(data))
    data
  })
  scores <- reactive({
    unknown <- setdiff(unique(responses()$question), key()$question)
    validate(need(
      length(unknown) == 0,
      paste("Questions missing from the key:", toString(unknown))
    ))
    score_responses(responses(), key(), case_sensitive = input$case_sensitive)
  })
  output$scores <- DT::renderDT({
    scores() |>
      mutate(passed = percent >= input$pass_mark) |>
      DT::datatable(rownames = FALSE, filter = "top") |>
      DT::formatStyle("passed", color = DT::styleEqual(c(TRUE, FALSE), c("green", "red")))
  })
  output$distribution <- renderPlot(plot_score_distribution(scores(), input$pass_mark))
  output$difficulty <- renderTable(item_difficulty(responses(), key()), digits = 2)
  output$download_scores <- downloadHandler(
    filename = function() paste0("quiz-scores-", Sys.Date(), ".csv"),
    content = function(file) readr::write_csv(scores(), file)
  )
}

shinyApp(ui, server)
