# models.py

from sentiment_data import *
from utils import *
from nltk import ngrams
from collections import Counter
from nltk.corpus import stopwords
import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

stemmer  = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # Converting input sentence to lower case for normalization
        stop_words = set(stopwords.words('english'))
        # removing 'not' from stopwords as it has its own importance in sentiments
        stop_words.remove('not')
        sentence = [word.lower() for word in sentence]
#         words =  [lemmatizer.lemmatize(token) for token in sentence]
        Phrase = [word for word in sentence if word not in stop_words]
#         Phrase = [stemmer.stem(word) for word in Phrase] 
        
        # Counter for the unigram features
        unigrams = Counter(Phrase)

        # new Counter to store the feature vector
        feature_vector_counter = Counter()
        # adding to indexer according to flag
        for word, count in unigrams.items():
            if add_to_indexer:
                # Getting the index for the word and adding it to the indexer if not present
                index = self.indexer.add_and_get_index(word)
            else:
                # Get the index from the indexer if it exists
                index = self.indexer.index_of(word)
                
                # Skip if the word is not in the indexer
                if index is None:
                    continue

            # Add the count of the word to the feature vector
            feature_vector_counter[index] += count
        
        return feature_vector_counter

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # Convert sentence to lower case for normalization
        stop_words = set(stopwords.words('english'))
        # removing 'not' from stopwords as it has its own importance in sentiments
        stop_words.remove('not')
        # pre processing
        Phrase = [word.lower() for word in sentence]
        Phrase =  [lemmatizer.lemmatize(token) for token in Phrase]
        Phrase = [word for word in Phrase if word not in stop_words]
        Phrase = [stemmer.stem(word) for word in Phrase] 

        # Counter for the bigram features
        bigram_counter = Counter(list(ngrams(Phrase, 2)))

        # Counter to store the feature vector
        feature_vector = Counter()

        for bigram, count in bigram_counter.items():
            bigram_sting = f"{bigram[0]}_{bigram[1]}"
            if add_to_indexer:
                # adding bigram index it to the indexer if not present
                index = self.indexer.add_and_get_index(bigram_sting)
            else:
                index = self.indexer.index_of(bigram_sting)

                # useful during testing
                if index is None:
                    continue

            # Add the count of the bigram to the feature vector
            feature_vector[index] += count
        
        return feature_vector

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer


    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # doing same pre processing steps like bigram and unigram but with some extra steps
        stop_words = set(stopwords.words('english'))
        stop_words.remove('not')
        Phrase = [word.lower() for word in sentence]
        Phrase =  [lemmatizer.lemmatize(token) for token in Phrase]
        Phrase = [word for word in Phrase if word not in stop_words]
        # stemmer is added here 
        Phrase = [stemmer.stem(word) for word in Phrase] 

        # making both unigram and bigrams
        unigram_counts = Counter(Phrase)
        bigram_counts = Counter(ngrams(Phrase, 2))

        # Counter to store the feature vector
        feature_vector = Counter()

        # Adding unigram features to the feature vector
        for word in unigram_counts:
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)

            if index is not None:
                feature_vector[index] += 1

        # Adding bigram features to the feature vector
        for bigram in bigram_counts:
            bigram_str = f"{bigram[0]}_{bigram[1]}"
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram_str)
            else:
                index = self.indexer.index_of(bigram_str)

            if index is not None:
                feature_vector[index] += 1  # Count occurrences

        return feature_vector



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: Counter, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        # Extract features from the sentence
        feature_vector = self.feat_extractor.extract_features(sentence, add_to_indexer=False)

        # finding scores
        score = sum(self.weights.get(feat, 0) * value for feat, value in feature_vector.items())

        # returning 1 if the score is positive, else 0
        return 1 if score > 0 else 0



class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    
    def __init__(self, weights: Counter, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        # Extract features from the sentence
        feature_vector = self.feat_extractor.extract_features(sentence, add_to_indexer=False)

        # finding score
        score = sum(self.weights.get(feat, 0) * value for feat, value in feature_vector.items())

        # Apply the sigmoid function
        probability = 1 / (1 + np.exp(-score))

        # returning 1 if the probability is > 0.5, else 0
        return 1 if probability > 0.5 else 0



def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    """

    # Initialize indexer 
    indexer = feat_extractor.get_indexer()
    
    # Precompute feature vectors for all training examples to avoid delay for each epoch
    # this also helps for finding the vocab size and to find size for weight vector
    feature_vectors = []
    for sent in train_exs:
        sentence = sent.words
        feature_vector = feat_extractor.extract_features(sentence, add_to_indexer=True)

        # Directly append the feature vector and label
        feature_vectors.append((feature_vector, sent.label))

    # size of the weight is the index size as all are examples are added
    weight_vector_size = len(indexer)
    # uniformly initializing weight vector
    weight_vector = np.random.uniform(low=-0.01, high=0.01, size=weight_vector_size)
    num_epochs =60
    learning_rate = 0.01

    # Training loop
    for epoch in range(1,num_epochs):
        # Shuffling examples each epoch
        np.random.shuffle(feature_vectors)  

        for feature_vector, true_label in feature_vectors:
            # finding score
            score = sum(weight_vector[feat] * value for feat, value in feature_vector.items() if feat < weight_vector_size)

            # Prediction
            prediction = 1 if score > 0 else 0

            # Update weights if the prediction is wrong
            if prediction != true_label:
                for feat, value in feature_vector.items():
                    if feat < weight_vector_size:
                        # added regularization
                        lambda_reg = 0.01
                        weight_vector[feat] += learning_rate * (true_label - prediction) * value - lambda_reg * weight_vector[feat]
        # learning rate decay
#         learning_rate*=0.9
    # Returning the trained PerceptronClassifier by converting the vectors to counters
    return PerceptronClassifier(Counter({i: weight_vector[i] for i in range(weight_vector_size) if weight_vector[i] != 0}), feat_extractor)



def get_top_pn_words(classifier, indexer, num_words=10):
    """Compute the top words function."""
    # Extract weights
    weights = classifier.weights

    # (weight, word) tuples
    word_pairs = []
    for idx, weight in weights.items():
        word = indexer.get_object(idx)
        word_pairs.append((weight, word))

    # Sort by weights
    # Highest positive weights
    word_pairs.sort(reverse=True, key=lambda x: x[0])  
    top_pos_wrds = word_pairs[:num_words]
    # Lowest negative weight
    word_pairs.sort(key=lambda x: x[0]) 
    top_neg_wrds = word_pairs[:num_words]
    return top_pos_wrds, top_neg_wrds



def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

log_likelihoods = []
weight_vectors=[]
def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with logistic regression.
    """

    # Initialize indexer 
    indexer = feat_extractor.get_indexer()

    dev_accuracies = []
    
    # Precompute feature vectors for all training examples to avoid delay for each epoch
    # this also helps for finding the vocab size and to find size for weight vector
    feature_vectors = []
    for sent in train_exs:
        sentence = sent.words
        feature_vector = feat_extractor.extract_features(sentence, add_to_indexer=True)

        # append the feature vector and label
        feature_vectors.append((feature_vector, sent.label))

    # size of the weight is the index size as all are examples are added
    weight_vector_size = len(indexer)
    weight_vector = np.random.uniform(low=-0.01, high=0.01, size=weight_vector_size)
    num_epochs = 70
    learning_rate = 0.01

    # Training loop
    for epoch in range(num_epochs):
        total_log_likelihood = 0.0
        # Shuffle training examples each epoch
        np.random.shuffle(feature_vectors)  

        for feature_vector, true_label in feature_vectors:
            # score finding
            score = sum(weight_vector[feat] * value for feat, value in feature_vector.items() if feat < weight_vector_size)
            
            # sigmoid function to get the predicted probability
            predicted_prob = sigmoid(score)

            # Prediction
            prediction = 1 if predicted_prob > 0.5 else 0
            # Update log likelihood
            total_log_likelihood += true_label * np.log(predicted_prob) + (1 - true_label) * np.log(1 - predicted_prob)

            # Update weights
            for feat, value in feature_vector.items():
                if feat < weight_vector_size:
                    weight_vector[feat] += learning_rate * (true_label - predicted_prob) * value
        
        # Record log likelihood
        log_likelihoods.append(total_log_likelihood)
        weight_vectors.append(weight_vector.copy())
        # Optionally, decrease the learning rate
#         learning_rate *= 0.9

    # Returning the trained PerceptronClassifier by converting the vectors to counters
    return PerceptronClassifier(Counter({i: weight_vector[i] for i in range(weight_vector_size) if weight_vector[i] != 0}), feat_extractor)




# import matplotlib.pyplot as plt

# def plot_and_save_log_likelihood_and_dev_accuracy(log_likelihoods, dev_accuracies):
#     """
#     Plots log likelihood and dev accuracy over iterations and saves the plot as a PNG image.
#     """
#     # Define iterations
#     iterations = range(len(log_likelihoods))

#     # Create a figure and axis objects
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     # Plot log likelihood on the primary y-axis
#     ax1.set_xlabel('Iterations')
#     ax1.set_ylabel('Log Likelihood', color='tab:blue')
#     ax1.plot(iterations, log_likelihoods, color='tab:blue', label='Log Likelihood')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     # Create a second y-axis for development accuracy
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Development Accuracy', color='tab:orange')
#     ax2.plot(iterations, dev_accuracies, color='tab:orange', label='Dev Accuracy')
#     ax2.tick_params(axis='y', labelcolor='tab:orange')

#     # Adding a title and grid
#     plt.title(f'Log Likelihood and Development Accuracy vs. Iterations (Learning Rate: {0.1})')
#     ax1.grid(True)

#     # Adjust layout and save plot to file
#     fig.tight_layout()
#     plt.savefig("filename_0.1.jpg")

#     # Close the plot to avoid display in certain environments
#     plt.close()

def calculate_accuracy(exs: List[SentimentExample], weight_vector: np.ndarray, feat_extractor: FeatureExtractor) -> float:
    correct = 0
    for ex in exs:
        feature_vector = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        score = sum(weight_vector[feat] * value for feat, value in feature_vector.items() if feat < len(weight_vector))
        prediction = 1 if sigmoid(score) > 0.5 else 0
        if prediction == ex.label:
            correct += 1
    return correct / len(exs)

def train_and_plot( dev_exs: List[SentimentExample], feat_extractor: FeatureExtractor, log_likelihoods, weight_vectors):
    plt.figure(figsize=(12, 8))
    num_epochs = len(log_likelihoods)
    lr=1
    dev_accuracies = [calculate_accuracy(dev_exs, w, feat_extractor) for w in weight_vectors]

#     plot_and_save_log_likelihood_and_dev_accuracy(log_likelihoods, dev_accuracies)

    
    
def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
#         top_p_words, top_n_words = get_top_pn_words(model, feat_extractor.get_indexer())

#         print("Top 10 positive weights:")
#         for weight, word in top_p_words:
#             print(f"{word}: {weight}")

#         print("\nTop 10 negative weights:")
#         for weight, word in top_n_words:
#             print(f"{word}: {weight}")
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)

#         train_and_plot(dev_exs, feat_extractor,  log_likelihoods, weight_vectors)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model


