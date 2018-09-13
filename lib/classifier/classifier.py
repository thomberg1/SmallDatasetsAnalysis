# Wrapper class to run a pytorch NN classifier in the sklearn framework.

# References
# http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
# https://github.com/dnouri/skorch

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import  check_array
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.model_selection import train_test_split


from .stopping import Stopping
from .error import InitializationFailedError

from .utilities import *

#######################################################################################################################

class Classifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper class for PyTorch models so that they can be used in sklearn especially GridSearch.
    """
    def __init__(self,
                 module,
                 module_D_in = 32,
                 module_D_out = 2,
                 module_H = 10 ,
                 module_dropout=0.5,
                 module_initialize = torch_weight_init ,
                 max_epochs = 1000,
                 batch_size = 40 ,
                 batch_shuffle = True,
                 validation_ratio = 0.2 ,
                 validation_shuffle = True,
                 optimizer = torch.optim.SGD ,
                 optimizer_lr = 0.1 ,
                 optimizer_momentum = 0.9 ,
                 optimizer_weight_decay = 5e-4 ,
                 optimizer_nesterov = True ,
                 criterion = nn.CrossEntropyLoss ,
                 criterion_size_average = True ,
                 scheduler = torch.optim.lr_scheduler.LambdaLR ,
                 scheduler_lr_lambda = exp_decay ,
                 stopping = Stopping ,
                 stopping_patience = 400 ,
                 scoring = metrics.f1_score ,
                 random_state = 10,
                 progress_bar = None,
                 verbose=0
                 ):

        self.module = module

        self.module_D_in = module_D_in
        self.module_D_out = module_D_out
        self.module_H = module_H
        self.module_dropout = module_dropout

        self.module_initialize = module_initialize

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.validation_ratio = validation_ratio
        self.validation_shuffle = validation_shuffle

        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.optimizer_momentum = optimizer_momentum
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_nesterov = optimizer_nesterov

        self.criterion = criterion
        self.criterion_size_average = criterion_size_average

        self.scheduler = scheduler
        self.scheduler_lr_lambda = scheduler_lr_lambda

        self.stopping = stopping
        self.stopping_patience = stopping_patience

        self.scoring = scoring

        self.random_state = random_state

        self.progress_bar = progress_bar

        self.verbose = verbose

        self.epoch_ = 0
        self.epochs_ = []
        self.loss_train_ = []
        self.lr_train_ = []
        self.score_train_ = []
        self.loss_valid_ = []
        self.score_valid_ = []

        self._initalize()

    def _initalize(self):

        if not self.random_state is None:
            self._set_random_state()

        try:
            p = self._get_params("module_")
            self.module_ = self.module(**p)
            if torch.cuda.is_available():
                self.module_.cuda()
        except Exception as err:
            raise InitializationFailedError("Couldn't initalize Module!", err)

        try:
            p = self._get_params("optimizer_")
            self.optimizer_ = self.optimizer(self.module_.parameters(), **p)
        except Exception as err:
            raise InitializationFailedError("Couldn't initalize Optimizer!", err)

        try:
            p = self._get_params("criterion_")
            self.criterion_ = self.criterion(**p)
        except Exception as err:
            raise InitializationFailedError("Couldn't initalize Criterion!", err)

        try:
            p = self._get_params("scheduler_")
            self.scheduler_ = self.scheduler(self.optimizer_, **p)
        except Exception as err:
            raise InitializationFailedError("Couldn't initalize Scheduler!", err)

        try:
            p = self._get_params("stopping_")
            self.stopping_ = self.stopping(self.module_, **p)
        except Exception as err:
            raise InitializationFailedError("Couldn't initalize Stopping!", err)

        try:
            self.scoring_ = self.scoring
        except Exception as err:
            raise InitializationFailedError("Couldn't initalize Scoring!", err)


    def _set_random_state(self):
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)

    def fit(self, X, y, sample_weight=None):
        """
        Fits the pytorch model to the given input variables X and dependent variables y.
        """
        X = check_array(X, accept_sparse=False)
        X, y = check_X_y(X, y)             # Check that X and y have correct shape
        self.classes_ = unique_labels(y)   # Store the classes seen during fit

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        self.X_, self.y_ = X, y

        self._fit(X, y, sample_weight)

        return self

    def _fit(self, X, y, sample_weight=None):

        self.stopping_.initalize()

        X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                              test_size=self.validation_ratio,
                                                              shuffle=self.validation_shuffle,
                                                              random_state=self.random_state)

        epoch_itr = range(self.epoch_ + 1, self.epoch_ + self.max_epochs + 1)
        if self.progress_bar:
            epoch_itr = self.progress_bar(epoch_itr)

        for epoch in epoch_itr:
            if self.progress_bar:
                epoch_itr.set_description('Epoch')

            self.epochs_.append(epoch)

            self.scheduler_.step()

            loss, score  = self._train(X_train, y_train)
            self.loss_train_.append(loss)
            self.score_train_.append(score)

            epoch_lr = [float(param_group['lr']) for param_group in self.optimizer_.param_groups][0]
            self.lr_train_.append(epoch_lr)

            loss, score  = self._score(X_valid, y_valid)
            self.loss_valid_.append(loss)
            self.score_valid_.append(score)

            if self.progress_bar:
                epoch_itr.set_postfix( loss=self.loss_train_[-1],
                                        lr=epoch_lr,
                                        best="%i" % self.stopping_.best_score_epoch + "/%.4f" % self.stopping_.best_score )

            if self.stopping_.step(epoch, self.loss_valid_[-1], self.score_valid_[-1]):
                self.epochs_ = self.epochs_[:self.stopping_.best_score_epoch]
                self.loss_train_ = self.loss_train_[:self.stopping_.best_score_epoch]
                self.score_train_ = self.score_train_[:self.stopping_.best_score_epoch]
                self.loss_valid_ = self.loss_valid_[:self.stopping_.best_score_epoch]
                self.score_valid_ = self.score_valid_[:self.stopping_.best_score_epoch]
                self.module_.load_state_dict(self.stopping_.best_score_state)
                if self.verbose:
                    print("Early stopping at epoch: %d, score %f" % (self.epoch_, self.score_valid_[-1]))
                break

            self.epoch_ = epoch

    def _train(self, X, y):

        self.module_.train()

        idx_batch, loss_avg, score_avg = 0, 0.0, 0.0
        for idx_batch, (X_batch, y_batch) in enumerate(mini_batch(X, y, self.batch_size, self.batch_shuffle)):

            if X_batch.shape[0] <= 1: # don't train on batch size 1; causes error with batch norm
                continue

            inputs, target = torch.from_numpy(X_batch).float(), torch.from_numpy(y_batch).long()
            inputs, target = to_var(inputs), to_var(target)

            outputs = self.module_(inputs)

            loss = self.criterion_(outputs, target)
            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()

            loss_avg += loss.item()

            y_pred = torch.max(outputs, 1)[1]
            score_avg += self.scoring_(y_batch, to_np(y_pred))

            del inputs, target, outputs

        return loss_avg / (idx_batch + 1), score_avg / (idx_batch + 1)

    def predict(self, X):
        """
        Predict using the fitted pytorch model
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X, accept_sparse=False)

        y_pred = self._predict(X)

        return y_pred

    def _predict(self, X):

        self.module_.eval()
        with torch.no_grad():

            y_pred = []
            for X_batch in batch(X , self.batch_size):
                inputs = torch.from_numpy(X_batch).float()
                inputs = to_var(inputs)

                outputs = self.module_(inputs)

                out = torch.max(outputs, 1)[1]
                y_pred.extend( to_np( out ))

                del inputs, outputs

        return np.array(y_pred)

    def predict_proba(self, X):
        """Return the log of probability estimates."""

        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        y_prob = self._predict_proba(X)

        return y_prob

    def _predict_proba(self, X):

        self.module_.eval()

        y_pred = []
        for X_batch in batch(X , self.batch_size):
            inputs = torch.from_numpy(X_batch).float()
            inputs = to_var(inputs)

            outputs = self.module_(inputs)

            y_pred.extend( to_np( outputs ))

            del inputs, outputs

        return np.array(y_pred)

    def score(self, X, y):
        """
        Scores the pytorch model using the scoring function given at class instantiation
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X, accept_sparse=False)
        X, y = check_X_y(X, y)             # Check that X and y have correct shape

        _, score = self._score(X, y)

        return score

    def _score(self, X, y):

        self.module_.eval()
        with torch.no_grad():

            loss_avg = 0
            score_avg = 0
            for idx_batch, (X_batch, y_batch) in enumerate(mini_batch(X, y, self.batch_size, self.batch_shuffle)):

                inputs, target = torch.from_numpy(X_batch).float(), torch.from_numpy(y_batch).long()
                inputs, target = to_var(inputs), to_var(target)

                outputs = self.module_(inputs)

                loss = self.criterion_(outputs, target)
                loss_avg += loss.item()

                y_pred = torch.max(outputs, 1)[1]
                score_avg += self.scoring_(y_batch, to_np(y_pred))

                del inputs, target, outputs

        return loss_avg / (idx_batch + 1), score_avg / (idx_batch + 1)

    def set_params(self, **parameters):
        BaseEstimator.set_params(self, **parameters)

        self._initalize()

        return self

    def get_params(self, deep=True):
        return BaseEstimator.get_params(self, deep)

    def _get_params(self, prefix):
        return { key[len(prefix):]: val for key, val in self.__dict__.items() if (len(key) > len(prefix)) and key.startswith(prefix) }
