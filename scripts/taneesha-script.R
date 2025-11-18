## this script contains functions for pre-processing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)
library(tidymodels)

### Question 1 


load("C:/Users/prasa/Desktop/ds/pstat197/197a/module-2-claims-data-table-7/data/claims-raw.RData")
View(claims_raw)

# function to parse html and clean text
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

claims_parsed <- parse_data(claims_raw)
claims_nlp <- nlp_fn(claims_parsed)

## add header
parse_fn_header <- function(.html){
  html <- read_html(.html) 
  
  header <- html %>%
    html_elements('h1, h2') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    str_squish()
  
  
  body <- html %>%  
    html_elements('p') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
  
  tibble(
    header = header,
    text_clean = body
  )
}

# function to apply to claims data
parse_data_header <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_header(text_tmp)) %>%
    unnest_wider(text_clean) %>%
    mutate(text_clean = str_c(header, text_clean, sep = " ")) %>% 
    select(-header)
  return(out)
}

nlp_fn_header <- function(parse_data_header.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

safe_parse <- purrr::possibly(parse_fn_header,
                              otherwise = tibble(header="", text_clean=""))

chunks <- split(claims_raw, ceiling(seq_len(nrow(claims_raw)) / 100))

library(dplyr)
library(purrr)
library(tidyr)

claims_parsed_header <- map_df(chunks, function(chunk) {
  chunk %>%
    rowwise() %>%
    mutate(text_clean = safe_parse(text_tmp)) %>%
    unnest_wider(text_clean) %>%
    mutate(text_clean = str_c(header, text_clean, sep = " ")) %>%
    select(-header)
})

claims_nlp_header <- nlp_fn(claims_parsed_header)
View(claims_nlp_header)

######-Fit PCA Logistic Regression Model (no headers)-##########
library(dplyr)
library(rsample)
library(yardstick)
library(broom)   # for augment()
library(modelr)

claims_subset <- claims_nlp %>%
  select(-.id) %>%
  mutate(class = ifelse(bclass == "Relevant claim content", 1, 0)) %>%
  select(-bclass)

set.seed(101422)
claims_split <- initial_split(claims_subset, prop = 0.8)
train_data <- training(claims_split)
View(train_data)
test_data  <- testing(claims_split)

X_train <- as.matrix(select(train_data, -class))
X_test  <- as.matrix(select(test_data, -class))

zero_var_cols <- apply(X_train, 2, function(col) var(col) == 0)
X_train_filtered <- X_train[, !zero_var_cols]
X_test_filtered  <- X_test[, !zero_var_cols]

pca_res <- prcomp(X_train_filtered, center = TRUE, scale. = FALSE)
cumvar <- cumsum(pca_res$sdev^2 / sum(pca_res$sdev^2))
n_pc <- which(cumvar >= 0.9)[1]

X_train_pca <- as.data.frame(pca_res$x[, 1:n_pc])
X_test_pca  <- as.data.frame(predict(pca_res, newdata = X_test_filtered)[, 1:n_pc])

X_train_pca$class <- train_data$class
X_test_pca$class  <- test_data$class

fit_pca <- glm(class ~ ., data = X_train_pca, family = "binomial")

test_preds <- augment(fit_pca, newdata = X_test_pca, type.predict = "response") %>%
  rename(pred_prob = .fitted) %>%
  mutate(
    estimate = factor(ifelse(pred_prob > 0.5, 1, 0), levels = c(0,1)),
    truth    = factor(class, levels = c(0,1))
  )

class_metrics <- metric_set(
  yardstick::accuracy,
  yardstick::sensitivity,
  yardstick::specificity,
  yardstick::f_meas
)

results <- test_preds %>%
  class_metrics(
    truth = truth,
    estimate = estimate,
    event_level = "second" 
  )

######-Fit PCA Logistic Regression Model (with headers)-##########
claims_subset <- claims_nlp_header %>%
  select(-.id) %>%
  mutate(class = ifelse(bclass == "Relevant claim content", 1, 0)) %>%
  select(-bclass)

set.seed(101422)
claims_split <- initial_split(claims_subset, prop = 0.8)
train_data <- training(claims_split)
View(train_data)
test_data  <- testing(claims_split)

X_train <- as.matrix(select(train_data, -class))
X_test  <- as.matrix(select(test_data, -class))

zero_var_cols <- apply(X_train, 2, function(col) var(col) == 0)
X_train_filtered <- X_train[, !zero_var_cols]
X_test_filtered  <- X_test[, !zero_var_cols]

pca_res <- prcomp(X_train_filtered, center = TRUE, scale. = FALSE)
cumvar <- cumsum(pca_res$sdev^2 / sum(pca_res$sdev^2))
n_pc <- which(cumvar >= 0.9)[1]

X_train_pca <- as.data.frame(pca_res$x[, 1:n_pc])
X_test_pca  <- as.data.frame(predict(pca_res, newdata = X_test_filtered)[, 1:n_pc])

X_train_pca$class <- train_data$class
X_test_pca$class  <- test_data$class

fit_pca <- glm(class ~ ., data = X_train_pca, family = "binomial")

test_preds <- augment(fit_pca, newdata = X_test_pca, type.predict = "response") %>%
  rename(pred_prob = .fitted) %>%
  mutate(
    estimate = factor(ifelse(pred_prob > 0.5, 1, 0), levels = c(0,1)),
    truth    = factor(class, levels = c(0,1))
  )

class_metrics <- metric_set(
  yardstick::accuracy,
  yardstick::sensitivity,
  yardstick::specificity,
  yardstick::f_meas
)

results_header <- test_preds %>%
  class_metrics(
    truth = truth,
    estimate = estimate,
    event_level = "second" 
  )

results
results_header

# Question 2 

## bigram
nlp_bigram_fn <- function(parse_data.out, top_n = 1500) {
  
  out <- parse_data.out %>% 
    unnest_tokens(
      bigram,
      text_clean,
      token = "ngrams",
      n = 2
    ) %>%
    filter(!str_detect(bigram, "[0-9]")) %>%
    count(.id, bclass, bigram, name = "n") %>%
    bind_tf_idf(term = bigram, document = .id, n = n)
  
  # keep only the top-N most important bigrams
  top_bigrams <- out %>%
    count(bigram, wt = n, name = "total") %>%
    slice_max(total, n = top_n) %>%
    pull(bigram)
  
  out <- out %>% filter(bigram %in% top_bigrams)
  
  # pivot_wider is now safe
  out <- out %>%
    pivot_wider(
      id_cols = c(".id", "bclass"),
      names_from = bigram,
      values_from = tf_idf,
      values_fill = 0
    )
  
  return(out)
}
claims_bigrams <- nlp_bigram_fn(claims_parsed_header)
View(claims_bigrams)

library(rsample)
claims_subset <- claims_bigrams %>%
  mutate(class = ifelse(bclass == "Relevant claim content", 1, 0)) %>%
  select(-bclass)

set.seed(101422)
claims_split <- initial_split(claims_subset, prop = 0.8)
train_data <- training(claims_split)
test_data  <- testing(claims_split)

X_train <- as.matrix(select(train_data, -class, -.id))
X_test  <- as.matrix(select(test_data, -class, -.id))

zero_var_cols <- apply(X_train, 2, function(col) var(col) == 0)
X_train_filtered <- X_train[, !zero_var_cols]
X_test_filtered  <- X_test[, !zero_var_cols]

pca_bigram <- prcomp(X_train_filtered, center = TRUE, scale. = FALSE)
cumvar <- cumsum(pca_bigram$sdev^2 / sum(pca_bigram$sdev^2))
n_pc <- which(cumvar >= 0.9)[1]

X_train_pca <- as.data.frame(pca_bigram$x[, 1:n_pc])
X_test_pca  <- as.data.frame(predict(pca_bigram, newdata = X_test_filtered)[, 1:n_pc])

# --- stops running 
word_log_odds <- log(test_preds$pred_prob / (1 - test_preds$pred_prob))
X_test_pca$word_log_odds <- word_log_odds
X_train_pca$word_log_odds <- log(predict(fit_pca, newdata = X_train_pca, type = "response") /
                                   (1 - predict(fit_pca, newdata = X_train_pca, type = "response")))

X_train_pca$class <- train_data$class
X_test_pca$class  <- test_data$class

# fit log regression
fit_bigram <- glm(class ~ ., data = X_train_pca, family = "binomial")

bigram_preds <- augment(fit_bigram, newdata = X_test_pca, type.predict = "response") %>%
  rename(pred_prob = .fitted) %>%
  mutate(
    estimate = factor(ifelse(pred_prob > 0.5, 1, 0), levels = c(0,1)),
    truth    = factor(class, levels = c(0,1))
  )

results_bigram <- bigram_preds %>%
  class_metrics(
    truth = truth,
    estimate = estimate,
    event_level = "second"
  )

results_bigram

