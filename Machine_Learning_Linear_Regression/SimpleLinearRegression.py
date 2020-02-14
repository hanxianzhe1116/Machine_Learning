import numpy as np
import sklearn.metrics as skm


class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "The size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        fenzi = 0.0
        fenmu = 0.0
        for x_i, y_i in zip(x_train, y_train):
            fenzi += (x_i - x_mean) * (y_i - y_mean)
            fenmu += (x_i - x_mean) ** 2
        self.a_ = fenzi / fenmu
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "Must fit before predict!"
        # print(np.array([self._predict(x) for x in x_predict]))
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_signle):
        return self.a_ * x_signle + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "The size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        fenzi = (x_train - x_mean).dot(y_train - y_mean)
        fenmu = (x_train - x_mean).dot(x_train - x_mean)
        # for x_i, y_i in zip(x_train, y_train):
        #     fenzi += (x_i - x_mean) * (y_i - y_mean)
        #     fenmu += (x_i - x_mean) ** 2
        self.a_ = fenzi / fenmu
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "Must fit before predict!"
        # print(np.array([self._predict(x) for x in x_predict]))
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_signle):
        return self.a_ * x_signle + self.b_

    def score(self, x_test, y_test):
        """根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return skm.r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression2()"