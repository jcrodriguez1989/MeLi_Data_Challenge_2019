library("caret")
library("dplyr")
library("fastrtext")
library("keras")
library("readr")

##### Make them parameters!
lang <- "spanish"
debug <- TRUE # debug or final submission?

tr_data <- read_csv(paste0("data/train_", lang, "_proc.csv"))
#####

# initialize tokenizer and fit
max_words <- 40000
tokenizer <- text_tokenizer(max_words) %>%
  fit_text_tokenizer(tr_data$title)

word_index <- tokenizer$word_index
(max_words <- length(word_index) + 1)
# [1] 34659
(num_classes <- length(unique(tr_data$category)) + 1)
# [1] 1063

word_seq_train <- tokenizer %>%
  texts_to_sequences(tr_data$title)

# the tokenizer should be used to process new data
save_text_tokenizer(tokenizer, paste0("models/tokenizer_", lang, ".cancu"))

rm(tokenizer)
(max_seq_len <- max(unlist(lapply(word_seq_train, length))))
# [1] 18

y_train_vec <- tr_data$category
y_train <- to_categorical(as.numeric(as.factor(y_train_vec)))
colnames(y_train) <- c("", sort(unique(y_train_vec)))
rm(tr_data)

word_seq_train <- pad_sequences(word_seq_train, maxlen = max_seq_len)

# embedding matrix

# to get word embeddings we will use fasttext Multi-lingual word vectors:
# Pre-trained models for 157 different languages
# https://fasttext.cc/docs/en/crawl-vectors.html
if (lang == "spanish") {
  ftext_file <- normalizePath("models/cc.es.300.bin")
} else {
  ftext_file <- normalizePath("models/cc.pt.300.bin")
}
ftext <- load_model(ftext_file)

embedding_matrix <- ftext %>%
  get_word_vectors(c("", names(word_index)))
(embed_dim <- ncol(embedding_matrix))
# [1] 300

rm(ftext)
rm(word_index)

if (debug) {
  # separate between train/test and validation sets (balanced by category)
  set.seed(8818)
  sets <- c(.85, .15) # 85% train/test, 15 val
  stopifnot(sum(sets) == 1)
  idxs <- by(seq_len(length(y_train_vec)), y_train_vec, function(x) {
    train <- sample(x, length(x) * sets[[1]])
    val <- setdiff(x, train)
    list(train = train, val = val)
  })

  # shuffle because keras uses the last ones from training_set as validation
  tr_idxs <- sample(unlist(lapply(idxs, function(x) x$train)))
  val_idxs <- sample(unlist(lapply(idxs, function(x) x$val)))

  word_seq_val <- word_seq_train[val_idxs, ]
  y_val <- y_train_vec[val_idxs]
  word_seq_train <- word_seq_train[tr_idxs, ]
  y_train <- y_train[tr_idxs, ]
  y_train_vec <- y_train_vec[tr_idxs]
  rm(list = c("tr_idxs", "val_idxs", "idxs"))
}

## Models

# 1D convnet
get_1d_convnet <- function() {
  model <- keras_model_sequential() %>%
    layer_embedding(
      input_dim = max_words, output_dim = embed_dim,
      weights = list(embedding_matrix),
      input_length = max_seq_len, trainable = FALSE
    ) %>%
    layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu", padding = "same") %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters = 64, kernel_size = 7, activation = "relu", padding = "same") %>%
    layer_global_max_pooling_1d() %>%
    layer_dense(num_classes, activation = "softmax")
  attr(model, "model_name") <- "1d_convnet"
  model
}

# GRU
get_gru <- function() {
  model <- keras_model_sequential() %>%
    layer_embedding(
      input_dim = max_words, output_dim = embed_dim,
      weights = list(embedding_matrix),
      input_length = max_seq_len, trainable = FALSE
    ) %>%
    layer_gru(
      units = embed_dim,
      dropout = 0.1,
      recurrent_dropout = 0.5,
      return_sequences = TRUE
    ) %>%
    layer_gru(
      units = 64, activation = "relu",
      dropout = 0.1,
      recurrent_dropout = 0.5
    ) %>%
    layer_dense(num_classes, activation = "softmax")
  attr(model, "model_name") <- "gru"
  model
}

# LStM
get_lstm <- function() {
  model <- keras_model_sequential() %>%
    layer_embedding(
      input_dim = max_words, output_dim = embed_dim,
      weights = list(embedding_matrix),
      input_length = max_seq_len, trainable = FALSE
    ) %>%
    layer_lstm(
      units = embed_dim,
      dropout = 0.1,
      recurrent_dropout = 0.5,
      return_sequences = TRUE
    ) %>%
    layer_lstm(
      units = 64, activation = "relu",
      dropout = 0.1,
      recurrent_dropout = 0.5
    ) %>%
    layer_dense(num_classes, activation = "softmax")
  attr(model, "model_name") <- "lstm"
  model
}

# bidirectional GRU
get_bidir_gru <- function() {
  model <- keras_model_sequential() %>%
    layer_embedding(
      input_dim = max_words, output_dim = embed_dim,
      weights = list(embedding_matrix),
      input_length = max_seq_len, trainable = FALSE
    ) %>%
    bidirectional(layer_gru(
      units = embed_dim,
      dropout = 0.1,
      recurrent_dropout = 0.5,
      return_sequences = TRUE
    )) %>%
    bidirectional(layer_gru(
      units = 64, activation = "relu",
      dropout = 0.1,
      recurrent_dropout = 0.5
    )) %>%
    layer_dense(num_classes, activation = "softmax")
  attr(model, "model_name") <- "bidir_gru"
  model
}

# bidirectional LStM
get_bidir_lstm <- function() {
  model <- keras_model_sequential() %>%
    layer_embedding(
      input_dim = max_words, output_dim = embed_dim,
      weights = list(embedding_matrix),
      input_length = max_seq_len, trainable = FALSE
    ) %>%
    bidirectional(layer_lstm(
      units = embed_dim,
      dropout = 0.1,
      recurrent_dropout = 0.5,
      return_sequences = TRUE
    )) %>%
    bidirectional(layer_lstm(
      units = 64, activation = "relu",
      dropout = 0.1,
      recurrent_dropout = 0.5
    )) %>%
    layer_dense(num_classes, activation = "softmax")
  attr(model, "model_name") <- "bidir_lstm"
  model
}

# Manually select one to train
model <- get_1d_convnet()
model <- get_gru()
model <- get_lstm()
model <- get_bidir_gru()
model <- get_bidir_lstm()

model %>% summary()

# Note:
# I have tried simple NN architectures, different parameters and structures 
# should be trained to check out best results.
# Also, it would be interesting to try a 1D CNN before the GRU or LStM
# layers. And maybe some advanced models with layer concatenation.

model %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = "accuracy"
  )

num_epochs <- 7
batch_size <- 256

history <- model %>%
  fit(
    word_seq_train, y_train,
    batch_size = batch_size, epochs = num_epochs,
    validation_split = 0.1, shuffle = TRUE,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss", patience = 3, min_delta = 0.0001
      )
    ),
    view_metrics = FALSE
  )

if (debug) {
  preds <- model %>%
    predict_classes(word_seq_val)
  preds <- colnames(y_train)[-1][preds]
  mean(y_val == preds) # accuracy
  y_val_fc <- factor(y_val, levels = unique(c(y_val, preds)))
  preds_fc <- factor(preds, levels = unique(c(y_val, preds)))
  cfmtrx <- confusionMatrix(data = preds_fc, reference = y_val_fc)
  mean(cfmtrx$byClass[, "Recall"], na.rm = TRUE) # bal accuracy
}

# Note:
# Using just rows with "reliable" label_quality keeps too few rows, and thus,
# there are categories with almost no samples, this is why balanced accucary
# gives low values compared to normal accuracy.

# Metrics:
# Only spanish and reliable titles

# Accuracy - balanced accuracy
# 1d convnet  [1] 0.8844591 [1] 0.6036911
# GRU         [1] 0.9103718 [1] 0.6673788
# LStM        [1] 0.9129226 [1] 0.6687646
# bidir GRU   [1] 0.9217221 [1] 0.7088714
# bidir LStM  

# Save the model
save_model_hdf5(model, paste0(
  "models/model_", lang, "_", attr(model, "model_name"), ".h5"
))

# Note:
# None of these models has shown overfitting, so we could improve metrics by 
# simply adding layers and make the layers bigger.
# Also, as no overfitting was found, that is why I have not added regularization
# or dropout (appart from the present in GRU and LStM layers).

# Note:
# Commonly, these kind of competitions are won by ensembling models, this is 
# allowed for MeLi Data Challenge, however, here we are not going to test 
# ensembles.
