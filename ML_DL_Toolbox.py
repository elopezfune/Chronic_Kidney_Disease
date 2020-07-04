import numpy as np   #Importing Numpy
import pandas as pd  #Importing Pandas
import os, logging, datetime
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import lightgbm as lgb
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


class ML_DL_Toolbox(BaseEstimator, RegressorMixin):
    '''Sci-kit learn wrapper for creating pseudo-lebeled estimators'''

    def __init__(self, seed=42, prob_threshold=0.9, num_folds=10, num_iterations=50):
        '''@seed: random number generator
           @prob_threshold: probability threshold to select pseudo labeled events
           @num_folds: number of folds for the cross validation analysis
           @num_iterations: number of iterations for labelling
           '''
        self.seed = seed
        self.prob_threshold = prob_threshold
        self.num_folds = num_folds
        self.num_iterations = num_iterations
            
            
    def get_params(self, deep=True):
        return {"seed": self.seed, "prob_threshold": self.prob_threshold, "num_folds": self.num_folds,
                "num_iterations": self.num_iterations}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def gets_best_model(self, X,target):
        best_classifiers=[]
        outer_cv = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=1)
        model_factory = [AdaBoostClassifier(), BaggingClassifier(), BayesianGaussianMixture(),
                         BernoulliNB(), CalibratedClassifierCV(), CatBoostClassifier(verbose=False),
                         DecisionTreeClassifier(), ExtraTreesClassifier(), GaussianMixture(),
                         GaussianNB(), GradientBoostingClassifier(), KNeighborsClassifier(),
                         LinearDiscriminantAnalysis(), LogisticRegression(max_iter=1000), LogisticRegressionCV(max_iter=1000),
                         MLPClassifier(), QuadraticDiscriminantAnalysis(),
                         RandomForestClassifier(), SGDClassifier()
                         ]
        logging.basicConfig(filename="ml_dl_toolbox_logfilename.log", level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        scoring=('accuracy', 'neg_mean_squared_error')
        try:
            for el in model_factory:
                el.seed = self.seed
                scores = cross_validate(el, X.drop(target,axis=1), X[target], cv=outer_cv, n_jobs=-1,scoring=scoring)
                scores = abs(np.sqrt(np.mean(scores['test_neg_mean_squared_error'])*-1))/np.mean(scores['test_accuracy'])    
                score_description = [el,'{el}'.format(el=el.__class__.__name__),"%0.5f" % scores]
                best_classifiers.append(score_description)
                best_model=pd.DataFrame(best_classifiers,columns=["Algorithm","Model","RMSE/Accuracy"]).sort_values("RMSE/Accuracy",axis=0, ascending=True)
                best_model=best_model.reset_index() 
        except OSError:
            logging.error('Check data structure')
        else:
            logging.info('Best fitting algorithm: '+ best_model["Model"][0]+" RMSE/Accuracy: "+ best_model["RMSE/Accuracy"][0])
            return best_model["Algorithm"][0]