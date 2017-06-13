import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
import numpy as np
from sklearn.cross_validation import KFold

def _refit_cv(X, y, model, n_folds, model_type, X_test=None):
    ntrain = X.shape[0]
    if X_test is not None:
        ntest = X_test.shape[0]
        if 'classification' in model_type:
            oof_test = np.zeros((ntest,))
            oof_test = oof_test[:, np.newaxis]
            oof_train = np.zeros((ntrain, len(np.unique(y)))) 
        else:
            oof_test = np.zeros((ntest,))
            oof_test_skf = np.empty((n_folds, ntest))
            oof_train = np.zeros((ntrain,))

    kf = KFold(n=ntrain, n_folds=n_folds)

    print("X size: {}".format(X.shape))
    print("y size: {}".format(y.shape))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = X[train_index]
        y_tr = y[train_index]
        x_te = X[test_index]
        print("train_index: {}".format(type(train_index)))
        print("x_tr size: {}".format(x_tr.shape))
        print("y_tr size: {}".format(y_tr.shape))
        print("x_te size: {}".format(x_te.shape))
        model.refit(x_tr, y_tr)
        if 'classification' in model_type:
            oof_train[test_index] = model.predict_proba(x_te)
            if X_test is not None:
                oof_test = oof_test + model.predict_proba(X_test) 
        else:
            oof_train[test_index] = model.predict(x_te)
            if X_test is not None:
                oof_test_skf[i, :] = model.predict(X_test)
    if X_test is not None:
        if 'classification' in model_type:
            oof_test = oof_test/n_folds
        else:
            oof_test[:] = oof_test_skf.mean(axis=0)
    else:
        oof_test = None
    return oof_train, oof_test

def main():
    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = \
        sklearn.cross_validation.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60, per_run_time_limit=30,
        tmp_folder='/tmp/autoslearn_sequential_example_tmp',
        output_folder='/tmp/autosklearn_sequential_example_out',
        # Do not construct ensembles in parallel to avoid using more than one
        # core at a time. The ensemble will be constructed after auto-sklearn
        # finished fitting all machine learning models.
        ensemble_size=0, delete_tmp_folder_after_terminate=False)
    automl.fit(X_train.copy(), y_train.copy(), dataset_name='digits')
    # This call to fit_ensemble uses all models trained in the previous call
    # to fit to build an ensemble which can be used with automl.predict()
    automl.fit_ensemble(y_train, ensemble_size=50)

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print(automl.sprint_statistics())
    print(y_test[:5])
    print("Accuracy score: ", sklearn.metrics.accuracy_score(y_test, predictions))
    automl.refit(X_train.copy(), y_train.copy())
    predictions1 = automl.predict(X_test)
    print("Accuracy score after refit: ", sklearn.metrics.accuracy_score(y_test, predictions1)) 
    oof_train, oof_test = _refit_cv(X_train, y_train, automl, 5, 'classification', X_test)
    print(oof_train.shape)
    print(oof_test.shape)
    print(oof_train[:5])
    oof_train_metric = sklearn.metrics.accuracy_score(y_train, np.argmax(oof_train, axis=1))
    print("oof train metric: {}".format(oof_train_metric))

    if oof_test is not None:
        oof_test_metric = sklearn.metrics.accuracy_score(y_test, np.argmax(oof_test, axis=1))
        print("average test metric after refit cv: {}".format(oof_test_metric))
    else:
        oof_test_metric = 'N/A'   

if __name__ == '__main__':
    main()
