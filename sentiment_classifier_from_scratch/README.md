# Sentiment Classification Project

## Overview
This project focuses on extracting features(unigram and bigram) and training classifiers(logistic and perceptron) to perform binary sentiment classification on movie review snippets. The main goal is to gain hands-on experience with the standard machine learning workflow, including data reading, training, testing, and feature design.

## Dataset
The dataset used for this project is derived from the movie review dataset of Socher et al. (2013). It includes sentiment-labeled sentences obtained from [Rotten Tomatoes](https://www.rottentomatoes.com/). The task simplifies the sentiment classification to a binary positive/negative classification, discarding neutral sentences.

The dataset consists of:
- **Labels**: 0 (negative) or 1 (positive)
- **Format**: Each line contains a label followed by a tab and then the sentence, which has been tokenized but not lowercased.

The data is split into:
- Training set
- Development (dev) set
- Blind test set (labels not provided)

## Getting Started
To get started, ensure you have Python 3.5+ installed on your system. It is recommended to install the Anaconda distribution for ease of package management.

### Prerequisites
Install the necessary packages using Anaconda:
- `numpy`
- `nltk`
- `spacy`

### Installation
You can download the necessary code and data from [this link](https://drive.google.com/file/d/1uHdcnBC3iaOnsKea-xHmWOD7oxT-nTdU/view?usp=sharing). 

### Running the Project
After downloading and extracting the files, navigate to the project directory. To confirm everything is working correctly, run the following command:

```bash
python sentiment_classifier.py --model TRIVIAL --no_run_on_test
