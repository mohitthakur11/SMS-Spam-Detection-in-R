# Load required packages
library(tidyverse)
library(tm)
library(SnowballC)
library(wordcloud)
library(caret)
library(e1071)

# Read and prepare data
sms_data <- read_csv(file.choose()) %>%
  mutate(type = factor(type))

# Create text corpus and clean it
create_clean_corpus <- function(text) {
  VCorpus(VectorSource(text)) %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, stopwords()) %>%
    tm_map(removePunctuation) %>%
    tm_map(stemDocument) %>%
    tm_map(stripWhitespace)
}

sms_corpus_clean <- create_clean_corpus(sms_data$text)

# Create document-term matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# Split data into training and testing sets
set.seed(123)  # for reproducibility
train_index <- createDataPartition(sms_data$type, p = 0.75, list = FALSE)
sms_dtm_train <- sms_dtm[train_index, ]
sms_dtm_test <- sms_dtm[-train_index, ]
sms_train_labels <- sms_data$type[train_index]
sms_test_labels <- sms_data$type[-train_index]

# Visualize word clouds
plot_wordcloud <- function(text, title) {
  wordcloud(text, max.words = 40, scale = c(3, 0.5), random.order = FALSE, 
            colors = brewer.pal(8, "Dark2"), main = title)
}

par(mfrow = c(1, 2))
plot_wordcloud(subset(sms_data, type == "spam")$text, "Spam")
plot_wordcloud(subset(sms_data, type == "ham")$text, "Ham")

# Create indicator features for frequent words
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[, sms_freq_words]

# Convert counts to binary indicators
convert_counts <- function(x) ifelse(x > 0, "Yes", "No")
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

# Train and evaluate Naive Bayes models
train_and_evaluate <- function(train_data, train_labels, test_data, test_labels, laplace = 0) {
  model <- naiveBayes(train_data, train_labels, laplace = laplace)
  predictions <- predict(model, test_data)
  confusionMatrix(predictions, test_labels)
}

# Model without Laplace smoothing
model_results <- train_and_evaluate(sms_train, sms_train_labels, sms_test, sms_test_labels)
print(model_results)

# Model with Laplace smoothing
model_results_laplace <- train_and_evaluate(sms_train, sms_train_labels, sms_test, sms_test_labels, laplace = 1)
print(model_results_laplace)