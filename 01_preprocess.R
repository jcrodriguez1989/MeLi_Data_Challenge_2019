library("dplyr")
library("readr")
# library("SnowballC") # if using wordStem
library("tidytext")
library("tm")

##### Make them parameters!
lang <- "spanish"
debug <- TRUE # debug or final submission?

tr_data <- read_csv("data/train.csv")
# saveRDS(tr_data, "data/train.rds")
# tr_data <- readRDS("data/train.rds")
#####

# add ID to each line
tr_data$ID <- seq_len(nrow(tr_data))

# table of quality by language
tr_data %>%
  group_by(language) %>%
  count(label_quality)
# language   label_quality       n
# <chr>      <chr>           <int>
# 1 portuguese reliable       693318
# 2 portuguese unreliable    9306682
# 3 spanish    reliable       490927
# 4 spanish    unreliable    9509073

# number of different categories
by(tr_data$category, tr_data$language, function(x) length(unique(x)))
# tr_data$language: portuguese
# [1] 1576
# -----------------------------------------------
#   tr_data$language: spanish
# [1] 1574

# number of rows in each category, by language
tr_data %>%
  group_by(language) %>%
  count(category, sort = TRUE) %>%
  tail(n = 20)
# language   category                                 n
# <chr>      <chr>                                <int>
# 1 spanish    WASTE_CONTAINERS                        80
# 2 portuguese WOOD_PROTECTIVE_PAINTS                  79
# 3 spanish    INDUSTRIAL_BLENDERS                     73
# 4 spanish    NAIL_POLISH_DRYER_SPRAYS                73
# 5 portuguese BOXING_SPEED_BAGS                       67
# 6 portuguese COLD_FOOD_AND_DRINK_VENDING_MACHINES    56
# 7 spanish    CHESS_CLOCKS                            53
# 8 spanish    FORCE_GAUGES                            53
# 9 spanish    POLY_MAILERS                            52
# 10 spanish    CHECKOUT_COUNTERS                       51
# 11 spanish    SCALE_RULERS                            49
# 12 spanish    COMMERCIAL_POPCORN_MACHINES             36
# 13 portuguese PACKAGING_CONTAINERS                    33
# 14 portuguese MARTIAL_ARTS_FOOT_GUARDS                30
# 15 portuguese COFFEE_VENDING_MACHINES                 29
# 16 portuguese HAMBURGER_FORMERS                       23
# 17 portuguese FIELD_HOCKEY_STICKS                     20
# 18 spanish    SNACK_HOLDERS                            9
# 19 spanish    ANTI_STATIC_PLIERS                       5
# 20 spanish    CARD_PAYMENT_TERMINALS                   2

# Note:
# there are categories with too few rows, so, creating a fully balanced data set
# would have too few rows in total. Maybe I could try creating a partially
# balanced one.

if (debug) {
  # if debug, then we train only with reliable titles.
  # Maybe, it would be better to just randomly sample.
  tr_data <- tr_data %>%
    filter(label_quality == "reliable")
}

# we will build one model for each language, so we select one
tr_data_sel <- tr_data %>%
  filter(language == lang)

rm(tr_data)

# tokenize titles.
# to lower case.
# remove special characters - except tilde, ñ, etc. -
# depending if we use (and which) pre-trained embedding models, then maybe we
# should remove non alphanumeric characters (tilde, ñ, etc.).
tr_data_sel_tok <- tr_data_sel %>%
  unnest_tokens(output = word, input = title)

# how many words each sentence has
tr_data_sel_tok %>%
  count(ID) %>%
  select(n) %>%
  summary()
# n
# Min.   : 1.000
# 1st Qu.: 6.000
# Median : 8.000
# Mean   : 7.544
# 3rd Qu.: 9.000
# Max.   :21.000

# according to MeLi site, titles can be up to 60 chars long, lets check
tr_data_sel$title %>%
  nchar() %>%
  summary()
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 3.00   38.00   51.00   46.62   58.00   60.00

# Note:
# As MeLi titles have at most 60 characters, and, in this dataset, in median
# 8 words, the title classification problem might be slightly different to a
# common text classification problem, maybe it could be something similar to
# tweet classification.

# remove stop words
tr_data_sel_tok <- tr_data_sel_tok %>%
  anti_join(tibble(word = stopwords(lang)))

# remove short words ( < 3 chars )
tr_data_sel_tok <- tr_data_sel_tok %>%
  filter(tr_data_sel_tok$word %>% nchar() > 2)

# lets see if there are extremely repeated words that wont improve
# classification, for example, "regalo", "promocion"
tr_data_sel_tok %>%
  count(word, sort = TRUE) %>%
  mutate(perc = n / nrow(tr_data_sel_tok)) %>%
  head(n = 20) %>%
  as.data.frame()
# word     n         perc
# 1       original 15786 0.0053476334
# 2            led 15776 0.0053442459
# 3            kit 12368 0.0041897587
# 4          reloj  7656 0.0025935311
# 5            luz  7642 0.0025887885
# 6          mujer  7580 0.0025677855
# 7         oferta  7576 0.0025664304
# 8         hombre  7460 0.0025271345
# 9            usb  7376 0.0024986788
# 10         envio  6998 0.0023706283
# 11        gratis  6833 0.0023147333
# 12         nuevo  6775 0.0022950853
# 13         negro  6428 0.0021775363
# 14         juego  6095 0.0020647299
# 15       samsung  6000 0.0020325479
# 16           100  5603 0.0018980609
# 17       campera  5577 0.0018892532
# 18          mini  5278 0.0017879646
# 19           pro  5270 0.0017852545
# 20         nueva  5233 0.0017727205

# Apparently, most of these words are informative, so lets keep them

# be careful with stemming, if we use pre-trained embedding models, then check
# out if words could be stemmed
# tr_data_sel_tok <- tr_data_sel_tok %>%
#   mutate(word = wordStem(word, language = lang))

# remove sparse terms?
# remove words that appear less than n = 5 times.
# Of course other values of n should be tested, and maybe make it a percentage
# of the total number of words.
sparse_words <- tr_data_sel_tok %>%
  count(word, sort = TRUE)
(sparse_words <- sparse_words %>%
  # filter(sparse_words$n < nrow(sparse_words) * 0.000025) %>%
  filter(sparse_words$n < 5))
# word          n
# <chr>     <int>
# 1 0,14          4
# 2 0,2           4
# 3 0,20hp        4
# 4 0,32          4
# 5 0,33          4
# 6 0,42          4
# 7 0,45x10mt     4
# 8 0,55          4
# 9 0,60cm        4
# 10 0,76          4
# # … with 133,548 more rows
tr_data_sel_tok <- tr_data_sel_tok %>%
  filter(!tr_data_sel_tok$word %in% sparse_words$word)

# Merge again each title and save it in a file.
tr_data_sel_proc <- tr_data_sel_tok %>%
  group_by(ID) %>%
  summarise(title = paste(word, collapse = " "))

stopifnot(all(
  tr_data_sel_proc$ID == tr_data_sel$ID[tr_data_sel$ID %in% tr_data_sel_proc$ID]
))

tr_data_sel_proc <- cbind(
  tr_data_sel_proc,
  tr_data_sel[tr_data_sel$ID %in% tr_data_sel_proc$ID, c(
    "label_quality", "language", "category"
  )]
)

head(tr_data_sel_proc)
# ID                                                       title label_quality language            category
# 1 268                              play station volante hooligans      reliable  spanish       GAME_CONSOLES
# 2 273                              pilas energizer max tira pilas      reliable  spanish      CELL_BATTERIES
# 3 288                   afeitadora electrica philips envio gratis      reliable  spanish    SHAVING_MACHINES
# 4 387      estufa calefactor volcan 2500 kcal 42512v salida mandy      reliable  spanish        HOME_HEATERS
# 5 450 reloj pared vox tronic blanco numeros 23cm garantia oficial      reliable  spanish         WALL_CLOCKS
# 6 473                          excellent perro adulto small breed      reliable  spanish CATS_AND_DOGS_FOODS

write_csv(tr_data_sel_proc, paste0("data/train_", lang, "_proc.csv"))
