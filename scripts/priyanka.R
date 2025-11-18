#####################-QUESTION 1-####################################################

####-Add Headers-############

## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

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

# function to parse html and clean text
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

claims_parsed_header <- parse_data_header(claims_raw)
claims_nlp_header <- nlp_fn(claims_parsed_header)

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

##################-Question 3-#################
library(reticulate)
library(keras)
library(tensorflow)
py_require("tensorflow")

set.seed(123)
partitions <- claims_parsed %>%
  initial_split(prop = 0.8)

train_data <- training(partitions)
test_data  <- testing(partitions)

# extract text and labels
train_text   <- train_data$text_clean
train_labels <- ifelse(train_data$bclass == "Relevant claim content", 1, 0)

test_text    <- test_data$text_clean
test_labels  <- ifelse(test_data$bclass == "Relevant claim content", 1, 0)

# create text preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(train_text)

# define neural network
model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(1, activation = 'sigmoid')  # binary output

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# train the model
history <- model %>%
  fit(
    x = train_text,
    y = train_labels,
    validation_split = 0.3,
    epochs = 5,
    batch_size = 32
  )

# predictions on test set
pred_probs <- model %>% predict(test_text) %>% as.numeric()

# convert probabilities to factor with levels 0 and 1
pred_classes <- factor(ifelse(pred_probs > 0.5, 1, 0), levels = c(0,1))
truth <- factor(test_labels, levels = c(0,1))

# compute metrics using the same class_metrics function as PCA logistic regression
class_metrics <- metric_set(
  yardstick::accuracy,
  yardstick::sensitivity,
  yardstick::specificity,
  yardstick::f_meas
)

nn_results <- tibble(
  truth = truth,
  estimate = pred_classes
) %>%
  class_metrics(
    truth = truth,
    estimate = estimate,
    event_level = "second"
  )

nn_results
