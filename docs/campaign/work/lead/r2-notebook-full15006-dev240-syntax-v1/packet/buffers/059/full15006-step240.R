#' @importFrom magrittr %>%
#' @importFrom rlang .data
#' @import shiny
#' @import shinydashboard

#' @name consortr
#'
#' @title Shiny app for generating consort flow diagrams and other types of flow diagrams
#'
#' @description A consort diagram graphically depicts the passage of participants through a randomized clinical trial.
#'   This app can be used to easily create consort diagrams, and to visualize any other process where criteria are applied
#'   in succession to a dataset and it is of interest to know how many rows of the dataset remain after the application of each criterion.
#' @references
#' Moher, Schulz and Altman (2001)
#' The CONSORT statement: revised recommendations for improving the quality of reports of parallel-group randomised trials.
#' \emph{Lancet} \bold{357}, 1191-94.
#' @return none
#' @export
consortr <- function() {
  ui <- dashboardPage(
    dashboardHeader(title = "consortr"),
    dashboardSidebar(
      width = 100,
      sidebarMenu(id = "menu", sidebarMenuOutput("menu"))
    ),
    dashboardBody(
      # from https://stackoverflow.com/questions/56965843/height-of-the-box-in-r-shiny
      tags$head(tags$script(
        '
      // Define function to set height of "map" and "map_container"
      setHeight = function() {
        var window_height = $(window).height();
        var header_height = $(".main-header").height();

        var boxHeight = window_height - header_height - 30;

        $("#diagram_container").height(boxHeight);
        $("#diagram").height(boxHeight - 20);
        $("#prune_diagram_container").height(boxHeight);
        $("#prune_diagram").height(boxHeight - 20);
      };

      // Set input$box_height when the connection is established
      $(document).on("shiny:connected", function(event) {
        setHeight();
      });

      // Refresh the box height on every window resize event    
      $(window).on("resize", function(){
        setHeight();
      });
    '
      )),
      tabItems(
        tabItem(
          tabName = "consort",
          # Boxes need to be put in a row (or column)
          fluidRow(
            #box(plotOutput("plot1", height = 250)),
            box(
              id = "diagram_container",
              DiagrammeR::grVizOutput("diagram"),
              width = 9
            ),
            # box(verbatimTextOutput("print")),
            box(
              #title = "Controls",
              #sliderInput("slider", "Number of observations:", 1, 100, 50),
              textInput(
                inputId = "textinput",
                label = "Rename node",
                value = ""
              ),
              uiOutput("radio"),
              uiOutput("selector"),
              width = 3
            )
          )
        ),
        tabItem(
          tabName = "prune",
          fluidRow(
            box(
              id = "prune_diagram_container",
              DiagrammeR::grVizOutput("prune_diagram"),
              width = 9
            ),
            box(
              checkboxInput("hide", label = "Hide pruned nodes", value = FALSE),
              width = 3
            )
          )
        ),
        tabItem(
          tabName = "import",
          fluidRow(
            box(fileInput(
              "filename",
              "Upload data file",
              accept = c(".csv", ".txt")
            ))
          ),
          fluidRow(
            box(title = "dataset", width = 12, tableOutput("contents"))
          )
        ),
        tabItem(
          tabName = "export",
          fluidRow(
            box(
              downloadButton("download_diagram", "Download diagram"),
              downloadButton("download_metadata", "Download diagram metadata")
            )
          )
        )
      )
    )
  )

  server <- function(input, output, session) {
    dat <- reactive({
      file <- input$filename
      ext <- tools::file_ext(file$datapath)
      req(file)
      validate(need(
        ext == "csv" | ext == "txt",
        "Please upload a csv or txt file"
      ))
      data.table::fread(file$datapath) |>
        as.data.frame()
    })

    output$menu <- renderMenu({
      if (is.null(input$filename)) {
        sidebarMenu(
          menuItem(
            "Import",
            tabName = "import",
            icon = icon("import", lib = "glyphicon")
          )
        )
      } else {
        sidebarMenu(
          menuItem(
            "Import",
            tabName = "import",
            icon = icon("import", lib = "glyphicon")
          ),
          menuItem(
            "Consort",
            tabName = "consort",
            icon = icon("sort-by-attributes", lib = "glyphicon")
          ),
          menuItem(
            "Prune",
            tabName = "prune",
            icon = icon("scissors", lib = "glyphicon")
          ),
          menuItem(
            "Export",
            tabName = "export",
            icon = icon("export", lib = "glyphicon")
          )
        )
      }
    })

    output$contents <- renderTable(dat())
    react_list <- reactiveValues()
    observeEvent(input$filename, {
      react_list$sel_node <- 1
      react_list$criteria <- Initialize(dat()) |>
        SelectNode(1)
    })
    observeEvent(input$textinput, {
      if (!input$textinput == "") {
        react_list$criteria <- Rename(
          react_list$criteria,
          react_list$sel_node,
          input$textinput
        )
      }
    })
    output$radio <- renderUI({
      # remove variables already split on above this node
      vars <- colnames(dat())
      parent_split_vars <- react_list$criteria$split_var[GetParents(
        react_list$criteria,
        react_list$sel_node
      )]
      var_choices <- setdiff(c(NO_SPLIT, vars), parent_split_vars)
      selected <- react_list$criteria$split_var[react_list$sel_node]
      radioButtons(
        "radio",
        "Split variable",
        choices = var_choices,
        selected = selected
      )
    })
    observeEvent(input$radio, {
      # remove child nodes
      # then, if needs to be split, split
      if (
        !input$radio %in% react_list$criteria$split_var[react_list$sel_node]
      ) {
        react_list$criteria <- DeleteChildren(
          react_list$criteria,
          react_list$sel_node
        )
        if (!input$radio == NO_SPLIT) {
          react_list$criteria <- AddChildren(
            react_list$criteria,
            react_list$sel_node,
            input$radio
          )
        }
      }
    })
    observeEvent(input$diagram_click, {
      sel_node <- as.numeric(gsub("node", "", input$diagram_click$id))
      react_list$sel_node <- sel_node
      react_list$criteria <- SelectNode(react_list$criteria, sel_node)
      updateTextInput(session = session, inputId = "textinput", value = "")
    })
    observeEvent(input$prune_diagram_click, {
      sel_node <- as.numeric(gsub("node", "", input$prune_diagram_click$id))
      react_list$criteria <- ToggleHidden(react_list$criteria, sel_node)
    })

    output$download_metadata <- downloadHandler(
      filename = function() {
        paste("diagram_metadata.csv", sep = "")
      },
      content = function(file) {
        to_write <- react_list$criteria
        to_write$data <- NULL
        utils::write.csv(x = to_write, file, row.names = FALSE)
      }
    )
    output$download_diagram <- downloadHandler(
      filename = function() {
        paste("diagram.pdf", sep = "")
      },
      content = function(file) {
        RenderCriteria(react_list$criteria, with_hiding = T) %>%
          DiagrammeR::export_graph(file_name = file)
        #RenderCriteria(react_list$criteria, with_hiding = T) %>%
        #  DiagrammeR::generate_dot() %>%
        #  grViz() %>%
        #  #grViz(diagram = "~/Desktop/dot_file_from_the_internet.dot") %>%
        #  export_svg %>%
        #  charToRaw %>%
        #  rsvg_pdf(file = file)
      }
    )

    output$diagram <- DiagrammeR::renderGrViz(DiagrammeR::grViz(DiagrammeR::generate_dot(RenderCriteria(
      react_list$criteria
    ))))
    output$prune_diagram <- DiagrammeR::renderGrViz(DiagrammeR::grViz(DiagrammeR::generate_dot(RenderCriteria(
      react_list$criteria,
      with_hiding = input$hide,
      with_hiding_colored = T
    ))))
  }

  shinyApp(ui, server)
}
