import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float("+inf")
        num_states = 1
        for n in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n)
            try:
                logL = model.score(self.X, self.lengths)
            except:
                continue
            '''
            The number of the parameter in the BIC selector is given by the sum of three terms:
            Initial state occupation probabilities = number of states
            Transition probabilities = number of states * (number of states - 1)
            Emission probabilities = number of states * number of features * 2 = numMeans+numCovars
            as discussed in the Udacity forum:
            https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/13
            '''
            p = n ** 2 + 2 * n * model.n_features - 1
            bic = - 2 * logL + p * math.log(sum(self.lengths))
            if bic < best_score:
                best_score = bic
                num_states = n
        if num_states > 1:
            return self.base_model(num_states)
        else:
            return None

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float("-inf")
        num_states = 1
        for n in range(self.min_n_components, self.max_n_components + 1):
            incr = 0
            anti_likelihood = 0
            model = self.base_model(n)
            for word in self.words:
                X, lengths = self.hwords[word]
                '''
                In the DIC selector the score of the "other words" needed to obtain the antilikelihood
                is computed on the model trained by the word on which we need to select the model.
                '''
                try:
                    logL = model.score(X, lengths)
                except:
                    continue
                if word == self.this_word:
                    likelihood = logL
                else:
                    anti_likelihood += logL
                    incr += 1
            if incr > 0:
                dic = likelihood - 1/incr * anti_likelihood
                if dic > best_score:
                    best_score = dic
                    num_states = n
        if num_states > 1:
            return self.base_model(num_states)
        else:
            return None

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        length = len(self.sequences)
        if length < 2:
            return None
        if length < 3:
            split = 2
        else:
            split = 3
        split_method = KFold(n_splits = split)
        best_score = float("-inf")
        num_states = 1
        for n in range(self.min_n_components, self.max_n_components + 1):
            average_likelihood = 0
            incr = 0
            '''
            The cross validation splits the data in two parts, training and test data.
            The score on the test data is evaluated and averaged to obtain the score of the word
            relative to the model.
            '''
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                model = self.base_model(n)
                X, lengths = combine_sequences(cv_test_idx, self.sequences)
                try:
                    logL = model.score(X, lengths)
                except:
                    continue
                average_likelihood += logL
                incr += 1
            if incr > 0:
                average_likelihood /= incr
                if average_likelihood > best_score:
                    num_states = n
                    best_score = average_likelihood
        if num_states > 1:
            return self.base_model(num_states)
        else:
            return None
