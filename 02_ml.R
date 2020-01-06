library("dplyr")
library("fastrtext")
library("h2o")
library("readr")

##### Make them parameters!
lang <- "spanish"
debug <- TRUE # debug or final submission?

tr_data <- read_csv(paste0("data/train_", lang, "_proc.csv"))
#####

# Note:
# Following the KISS principle, we will start this this classification problem
# with Machine Learning approaches. If the results are not that good, at least,
# we could get a baseline ;)

# to get word/sentence embeddings we will use fasttext Multi-lingual word
# vectors: Pre-trained models for 157 different languages
# https://fasttext.cc/docs/en/crawl-vectors.html
if (lang == "spanish") {
  ftext_file <- normalizePath("models/cc.es.300.bin")
} else {
  ftext_file <- normalizePath("models/cc.pt.300.bin")
}
ftext <- load_model(ftext_file)

# for each sentence get its fasttext vector representation
data_mtrx <- ftext %>%
  get_sentence_representation(tr_data$title) %>%
  data.frame(category = tr_data$category, .)

if (debug) {
  # separate between train/test and validation sets (balanced by category)
  set.seed(8818)
  sets <- c(.85, .15) # 85% train/test, 15 val
  stopifnot(sum(sets) == 1)
  idxs <- by(seq_len(nrow(data_mtrx)), data_mtrx$category, function(x) {
    train <- sample(x, length(x) * sets[[1]])
    val <- setdiff(x, train)
    list(train = train, val = val)
  })

  train_data <- data_mtrx[unlist(lapply(idxs, function(x) x$train)), ]
  val_data <- data_mtrx[unlist(lapply(idxs, function(x) x$val)), ]
} else {
  train_data <- data_mtrx
}

rm(list = setdiff(ls(), c(
  "debug", "lang", "train_data", "val_data"
)))

# start modeling with h2o
# h2o.shutdown(prompt = FALSE)
h2o.init(min_mem_size = "10G") # should have 10GB RAM free

train_data <- as.h2o(train_data)
if (debug) {
  val_data <- as.h2o(val_data)
}

# autoML
aml <- h2o.automl(
  y = "category",
  training_frame = train_data,
  max_runtime_secs = 60 * 60, # one hour train
  max_models = 10,
  seed = 8818
)
aml@leaderboard

# Note:
# It did not want to run on my laptop, too many categories for h2o ML models.
