aplos_download_results <- function(url, token, execution_id) {
  if (missing(url) || missing(token) || missing(execution_id)) {
    stop("All parameters (url, token, execution_id) are required.")
  }

  tenant_id <- AzureAuth::decode_jwt(
    token
  )$payload$`custom:aplos_user_tenant_id`
  user_id <- AzureAuth::decode_jwt(token)$payload$`custom:aplos_user_id`
  headers <- c(
    "Content-Type" = "application/json",
    "Authorization" = paste0("Bearer ", token)
  )

  # Get output file ID
  response <- tryCatch(
    {
      httr::GET(
        paste0(
          url,
          "/v3/tenants/",
          tenant_id,
          "/users/",
          user_id,
          "/executions/",
          execution_id,
          "/outputs/package"
        ),
        httr::add_headers(.headers = headers)
      )
    },
    error = function(e) stop("Output package request failed: ", e$message)
  )

  result <- jsonlite::fromJSON(httr::content(
    response,
    "text",
    encoding = "UTF-8"
  ))
  output_id <- result$data$id

  # Get download URL
  response <- tryCatch(
    {
      httr::GET(
        paste0(
          url,
          "/v3/tenants/",
          tenant_id,
          "/users/",
          user_id,
          "/files/",
          output_id,
          "/download-url"
        ),
        httr::add_headers(.headers = headers)
      )
    },
    error = function(e) stop("Download URL request failed: ", e$message)
  )

  result <- jsonlite::fromJSON(httr::content(
    response,
    "text",
    encoding = "UTF-8"
  ))
  df <- data.frame(
    url = result$data$downloadUrl,
    filename = result$data$fileName
  )
  return(df)
}