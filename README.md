# SMS Spam Detection using Naive Bayes Algorithm

## Dataset
You can download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) or find it in this folder.

## Problem Overview
This is a classification problem. While there are many algorithms to solve it, this project focuses on:
- Naive Bayes
- Support Vector Machine (SVM)

## Methodology
1. Convert message text into tokens
2. Apply Naive Bayes algorithm

## Required Libraries
The following R libraries are used in this project:

```r
library(NLP)          # Natural Language Processing
library(tm)           # Text Mining
library(SnowballC)    # Provides wordstem() function
library(wordcloud)    # Visualizing text data as a word cloud
library(e1071)        # Naive Bayes and SVM implementation
library(gmodels)      # Model evaluation tools
