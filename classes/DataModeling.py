from copy import deepcopy

from matplotlib.figure import Figure
from matplotlib.pyplot import figure, savefig, show, subplots
from typing import Literal
from numpy import array, ndarray, std, argsort, arange, mean
from pandas import Series
from sklearn.base import RegressorMixin
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA

from torch import no_grad, tensor
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from config.dslabs_functions import plot_bar_chart, \
    plot_multiline_chart, plot_horizontal_bar_chart, HEIGHT, plot_line_chart, plot_multibar_chart, \
    plot_confusion_matrix, \
    CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_forecasting_eval, plot_forecasting_series, \
    FORECAST_MEASURES


class DataModeling:

    def __init__(self, data_loader, X_train, X_test, y_train, y_test):
        self.data_loader = data_loader
        self.target = self.data_loader.target
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.squeeze()
        self.y_test = y_test.squeeze()

    def plot_evaluation_results(self, model, trn_y, prd_trn, tst_y, prd_tst, labels: ndarray, dataset) -> ndarray:
        evaluation: dict = {}

        # Calculate metrics for both train and test sets
        for key in CLASS_EVAL_METRICS:
            evaluation[key] = [
                CLASS_EVAL_METRICS[key](trn_y, prd_trn),  # Train metric
                CLASS_EVAL_METRICS[key](tst_y, prd_tst),  # Test metric
            ]

        # Prepare the parameters string for the plot title
        params_st: str = "" if () == model.get("params", ()) else str(model["params"])

        # Create subplots for evaluation metrics and confusion matrix
        fig: Figure
        axs: ndarray
        fig, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
        fig.suptitle(f'Best {model["metric"]} for {model["name"]} {params_st} - {dataset}')

        # Plot metrics as a bar chart
        plot_multibar_chart(["Train", "Test"], evaluation, ax=axs[0], percentage=True)

        # Compute and plot confusion matrix
        cnf_mtx_tst: ndarray = confusion_matrix(tst_y, prd_tst, labels=labels)
        plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1])

        # Return the evaluation results
        return evaluation

    def naive_Bayes_study(
            self, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
    ) -> tuple:
        estimators: dict = {
            "GaussianNB": GaussianNB(),
            #"MultinomialNB": MultinomialNB(),
            "BernoulliNB": BernoulliNB(),
        }

        xvalues: list = []
        yvalues: list = []
        best_model = None
        best_params: dict = {"name": "", "metric": metric, "params": ()}
        best_performance = 0
        for clf in estimators:
            xvalues.append(clf)
            estimators[clf].fit(trnX, trnY)
            prdY: array = estimators[clf].predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params["name"] = clf
                best_params[metric] = eval
                best_model = estimators[clf]
            yvalues.append(eval)
            # print(f'NB {clf}')
        plot_bar_chart(
            xvalues,
            yvalues,
            title=f"Naive Bayes Models ({metric}) - {self.data_loader.file_tag}",
            ylabel=metric,
            percentage=True,
        )

        return best_model, best_params

    def naive_bayes_classification(self):
        """
        Evaluate Naive Bayes models for both accuracy and recall and generate plots for each metric.
        Returns the results for both evaluations.
        """
        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values
        labels: list = list(self.y_train.unique())
        labels.sort()

        print(f"Train#={len(trnX)} Test#={len(tstX)}")

        # Accuracy
        # figure(figsize=(7, 5))
        # best_model, params= self.naive_Bayes_study(trnX, trnY, tstX, tstY, metric="accuracy")
        # savefig(f"graphs/classification/data_modeling/nb/{self.data_loader.file_tag}_nb_accuracy_study.png")
        # show()

        # Recall
        figure(figsize=(7, 5))
        best_model, params= self.naive_Bayes_study(trnX, trnY, tstX, tstY, metric="recall")
        savefig(f"graphs/classification/data_modeling/nb/{self.data_loader.file_tag}_nb_recall_study.png")
        show()

        # Print model performance
        print(f"\nModel Performance:")
        print(f"Best Model: {params['name']} | Metric: {params['metric']} | Score: {params[params['metric']]}")

        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)

        # Plot evaluation results
        figure()
        evaluation_results = self.plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(
            f'graphs/classification/data_modeling/nb/{self.data_loader.file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
        show()

        return evaluation_results

    def knn_study(
            self, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, dataset, k_max: int = 19, lag: int = 2, metric='accuracy'
    ) -> tuple[KNeighborsClassifier | None, dict]:
        dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']

        kvalues: list[int] = [i for i in range(1, k_max + 1, lag)]
        best_model: KNeighborsClassifier | None = None
        best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
        best_performance: float = 0.0

        values: dict[str, list] = {}
        for d in dist:
            y_tst_values: list = []
            for k in kvalues:
                clf = KNeighborsClassifier(n_neighbors=k, metric=d)
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params['params'] = (k, d)
                    best_params[metric] = eval
                    best_model = clf
                # print(f'KNN {d} k={k}')
            values[d] = y_tst_values
        print(f"KNN best with k={best_params['params'][0]} and {best_params['params'][1]}")
        plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric}) - {dataset}', xlabel='k', ylabel=metric,
                             percentage=True)

        return best_model, best_params

    def knn_classification(self, k_max: int = 25, lag: int = 2) -> dict:
        """
        Evaluate KNN models for both accuracy and recall and generate plots for each metric.
        Returns the results for both evaluations.
        """
        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values
        labels: list = list(self.y_train.unique())
        labels.sort()

        print(f"Train#={len(trnX)} Test#={len(tstX)}")

        # Accuracy Study
        # figure()
        # best_model, params = self.knn_study(
        #     trnX, trnY, tstX, tstY, self.data_loader.file_tag, k_max=k_max, lag=lag, metric="accuracy"
        # )
        # savefig(f"graphs/classification/data_modeling/knn/{self.data_loader.file_tag}_knn_accuracy_study.png")
        # show()

        # Recall Study
        figure()
        best_model, params = self.knn_study(
            trnX, trnY, tstX, tstY, self.data_loader.file_tag, k_max=k_max, lag=lag, metric="recall"
        )
        savefig(f"graphs/classification/data_modeling/knn/{self.data_loader.file_tag}_knn_recall_study.png")
        show()

        # Print model performance
        print("\nModel Performance:")
        print(f"Best Model: {params['name']} | Metric: {params['metric']} | Score: {params[params['metric']]}")

        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)

        figure()
        evaluation_results = self.plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(
            f'graphs/classification/data_modeling/knn/{self.data_loader.file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png'
        )
        show()

        # Overfitting Study
        distance = params["params"][1]
        K_MAX = 25
        kvalues = [i for i in range(1, K_MAX, 2)]
        y_tst_values = []
        y_trn_values = []

        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
            clf.fit(trnX, trnY)
            prd_tst_Y: array = clf.predict(tstX)
            prd_trn_Y: array = clf.predict(trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[params["metric"]](tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[params["metric"]](trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            kvalues,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"KNN Overfitting Study for {distance} - {self.data_loader.file_tag}",
            xlabel="K",
            ylabel=params["metric"],
            percentage=True,
        )
        savefig(f"graphs/classification/data_modeling/knn/{self.data_loader.file_tag}_knn_overfitting.png")
        show()

        return evaluation_results

    def trees_study(
            self, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, dataset, d_max: int = 10, lag: int = 2, metric='accuracy'
    ) -> tuple:
        criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
        depths: list[int] = [i for i in range(2, d_max + 1, lag)]

        best_model: DecisionTreeClassifier | None = None
        best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
        best_performance: float = 0.0

        values: dict = {}
        for c in criteria:
            y_tst_values: list[float] = []
            for d in depths:
                clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params['params'] = (c, d)
                    best_params[metric] = eval
                    best_model = clf
                # print(f'DT {c} and d={d}')
            values[c] = y_tst_values
        print(f"DT best with {best_params['params'][0]} and d={best_params['params'][1]}")
        plot_multiline_chart(depths, values, title=f'DT Models ({metric}) - {dataset}', xlabel='d', ylabel=metric, percentage=True)

        return best_model, best_params

    def decision_tree_classification(self):

        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values
        labels: list = list(self.y_train.unique())
        labels.sort()
        vars = self.X_train.columns.to_list()

        print(f'Train#={len(trnX)} Test#={len(tstX)}')

        # Accuracy Study
        # figure()
        # best_model, params = self.trees_study(trnX, trnY, tstX, tstY, self.data_loader.file_tag, d_max=25,
        #                                       metric='accuracy')
        # savefig(f'graphs/classification/data_modeling/dt/{self.data_loader.file_tag}_dt_accuracy_study.png')
        # show()

        # Recall Study
        figure()
        best_model, params = self.trees_study(trnX, trnY, tstX, tstY, self.data_loader.file_tag, d_max=25,
                                              metric='recall')
        savefig(f'graphs/classification/data_modeling/dt/{self.data_loader.file_tag}_dt_recall_study.png')
        show()

        # Print model performance
        print("\nModel Performance:")
        print(f"Best Model: {params['name']} | Metric: {params['metric']} | Score: {params[params['metric']]}")

        # Best Model Performance Analysis
        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)

        figure()
        evaluation_results = self.plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(
            f'graphs/classification/data_modeling/dt/{self.data_loader.file_tag}_dt_{params["name"]}_best_{params["metric"]}_eval.png')
        show()

        # Variables Importance
        tree_filename: str = f'graphs/classification/data_modeling/dt/{self.data_loader.file_tag}_dt_{params["metric"]}_best_tree'
        max_depth2show = 3
        st_labels: list[str] = [str(value) for value in labels]

        figure(figsize=(14, 6))
        plot_tree(
            best_model,
            max_depth=max_depth2show,
            feature_names=vars,
            class_names=st_labels,
            filled=True,
            rounded=True,
            impurity=False,
            precision=2,
        )
        savefig(tree_filename + ".png")

        importances = best_model.feature_importances_
        indices: list[int] = argsort(importances)[::-1]
        elems: list[str] = []
        imp_values: list[float] = []
        for f in range(len(vars)):
            elems += [vars[indices[f]]]
            imp_values += [importances[indices[f]]]
            print(f"{f + 1}. {elems[f]} ({importances[indices[f]]})")

        figure(figsize=(11, 11))
        plot_horizontal_bar_chart(
            elems,
            imp_values,
            title="Decision Tree variables importance - " + self.data_loader.file_tag,
            xlabel="importance",
            ylabel="variables",
            percentage=True,
        )
        savefig(f'graphs/classification/data_modeling/dt/{self.data_loader.file_tag}_dt_{params["metric"]}_vars_ranking.png')

        # Overfitting Study
        crit: Literal["entropy", "gini"] = params["params"][0]
        d_max = 25
        depths: list[int] = [i for i in range(2, d_max + 1, 1)]
        y_tst_values: list[float] = []
        y_trn_values: list[float] = []
        acc_metric = "recall"
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0)
            clf.fit(trnX, trnY)
            prd_tst_Y: array = clf.predict(tstX)
            prd_trn_Y: array = clf.predict(trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            depths,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"DT overfitting study for {crit} - {self.data_loader.file_tag}",
            xlabel="max_depth",
            ylabel=str(params["metric"]),
            percentage=True,
        )
        savefig(f'graphs/classification/data_modeling/dt/{self.data_loader.file_tag}_dt_{params["metric"]}_overfitting.png')
        show()

        return evaluation_results

    def mlp_study(
            self, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array,
            nr_max_iterations: int = 2500, lag: int = 500, metric: str = "accuracy",
    ) -> tuple[MLPClassifier | None, dict]:
        nr_iterations: list[int] = [lag] + [
            i for i in range(2 * lag, nr_max_iterations + 1, lag)
        ]

        lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
            "constant",
            "invscaling",
            "adaptive",
        ]  # only used if optimizer='sgd'
        learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

        best_model: MLPClassifier | None = None
        best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
        best_performance: float = 0.0

        values: dict = {}
        _, axs = subplots(
            1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
        )
        for i in range(len(lr_types)):
            type: str = lr_types[i]
            values = {}
            for lr in learning_rates:
                warm_start: bool = False
                y_tst_values: list[float] = []
                for j in range(len(nr_iterations)):
                    clf = MLPClassifier(
                        learning_rate=type,
                        learning_rate_init=lr,
                        max_iter=lag,
                        warm_start=warm_start,
                        activation="logistic",
                        solver="sgd",
                        verbose=False,
                    )
                    clf.fit(trnX, trnY)
                    prdY: array = clf.predict(tstX)
                    eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                    y_tst_values.append(eval)
                    warm_start = True
                    if eval - best_performance > DELTA_IMPROVE:
                        best_performance = eval
                        best_params["params"] = (type, lr, nr_iterations[j])
                        best_params[metric] = eval
                        best_model = clf
                    # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
                values[lr] = y_tst_values
            plot_multiline_chart(
                nr_iterations,
                values,
                ax=axs[0, i],
                title=f"MLP with {type}",
                xlabel="nr iterations",
                ylabel=metric,
                percentage=True,
            )
        print(
            f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
        )

        return best_model, best_params

    def mlp_classification(self):

        LAG: int = 500
        NR_MAX_ITER: int = 5000
        eval_metric = "recall"

        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values
        labels: list = list(self.y_train.unique())
        labels.sort()
        vars = self.X_train.columns.to_list()

        print(f'Train#={len(trnX)} Test#={len(tstX)}')
        print(f'Labels={labels}')

        # Recall Study
        figure()
        best_model, params = self.mlp_study(trnX, trnY, tstX, tstY, nr_max_iterations=NR_MAX_ITER, lag=LAG, metric=eval_metric)
        savefig(f"graphs/classification/data_modeling/mlp/{self.data_loader.file_tag}_mlp_{eval_metric}_study.png")
        show()

        print(f"Best Model: {params['name']} | Metric: {params['metric']} | Score: {params[params['metric']]}")

        # Best Model Performance Analysis
        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)
        figure()
        evaluation_results = self.plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(f'graphs/classification/data_modeling/mlp/{self.data_loader.file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png')
        show()

        # Overfitting Study
        lr_type: Literal["constant", "invscaling", "adaptive"] = params["params"][0]
        lr: float = params["params"][1]

        nr_iterations: list[int] = [LAG] + [i for i in range(2 * LAG, NR_MAX_ITER + 1, LAG)]

        y_tst_values: list[float] = []
        y_trn_values: list[float] = []
        acc_metric = "accuracy"

        warm_start: bool = False
        for n in nr_iterations:
            clf = MLPClassifier(
                warm_start=warm_start,
                learning_rate=lr_type,
                learning_rate_init=lr,
                max_iter=n,
                activation="logistic",
                solver="sgd",
                verbose=False,
            )
            clf.fit(trnX, trnY)
            prd_tst_Y: array = clf.predict(tstX)
            prd_trn_Y: array = clf.predict(trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
            warm_start = True

        figure()
        plot_multiline_chart(
            nr_iterations,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
            xlabel="nr_iterations",
            ylabel=str(eval_metric),
            percentage=True,
        )
        savefig(f"graphs/classification/data_modeling/mlp/{self.data_loader.file_tag}_mlp_{eval_metric}_overfitting.png")

        # Loss Study
        figure()
        plot_line_chart(
            arange(len(best_model.loss_curve_)),
            best_model.loss_curve_,
            title="Loss curve for MLP best model training",
            xlabel="iterations",
            ylabel="loss",
            percentage=False,
        )
        savefig(f"graphs/classification/data_modeling/mlp/{self.data_loader.file_tag}_mlp_{eval_metric}_loss_curve.png")

        return evaluation_results

    def random_forests_study(self,
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array,
        nr_max_trees: int = 2500, lag: int = 500, metric: str = "accuracy",
    ) -> tuple[RandomForestClassifier | None, dict]:
        n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
        max_depths: list[int] = [2, 5, 7]
        max_features: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

        best_model: RandomForestClassifier | None = None
        best_params: dict = {"name": "RF", "metric": metric, "params": ()}
        best_performance: float = 0.0

        values: dict = {}

        cols: int = len(max_depths)
        _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
        for i in range(len(max_depths)):
            d: int = max_depths[i]
            values = {}
            for f in max_features:
                y_tst_values: list[float] = []
                for n in n_estimators:
                    clf = RandomForestClassifier(
                        n_estimators=n, max_depth=d, max_features=f
                    )
                    clf.fit(trnX, trnY)
                    prdY: array = clf.predict(tstX)
                    eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                    y_tst_values.append(eval)
                    if eval - best_performance > DELTA_IMPROVE:
                        best_performance = eval
                        best_params["params"] = (d, f, n)
                        best_params[metric] = eval
                        best_model = clf
                    # print(f'RF d={d} f={f} n={n}')
                values[f] = y_tst_values
            plot_multiline_chart(
                n_estimators,
                values,
                ax=axs[0, i],
                title=f"Random Forests with max_depth={d}",
                xlabel="nr estimators",
                ylabel=metric,
                percentage=True,
            )
        print(f'RF best for {best_params["params"][2]} trees (d={best_params["params"][0]} and f={best_params["params"][1]})')
        return best_model, best_params

    def random_forest_classification(self):
        eval_metric = "recall"

        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values
        labels: list = list(self.y_train.unique())
        labels.sort()
        vars = self.X_train.columns.to_list()

        print(f"Train#={len(trnX)} Test#={len(tstX)}")
        print(f"Labels={labels}")

        # Recall Study
        figure()
        best_model, params = self.random_forests_study(
            trnX,
            trnY,
            tstX,
            tstY,
            nr_max_trees=1000,
            lag=250,
            metric=eval_metric,
        )
        savefig(f"graphs/classification/data_modeling/rf/{self.data_loader.file_tag}_rf_{eval_metric}_study.png")
        show()

        print(f"Best Model: {params['name']} | Metric: {params['metric']} | Score: {params[params['metric']]}")

        # Best Model Performance Analysis
        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)
        figure()
        evaluation_results = self.plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(f'graphs/classification/data_modeling/rf/{self.data_loader.file_tag}_rf_{params["name"]}_best_{params["metric"]}_eval.png')
        show()

        # Variables Importance Study
        stdevs: list[float] = list(
            std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
        )
        importances = best_model.feature_importances_
        indices: list[int] = argsort(importances)[::-1]
        elems: list[str] = []
        imp_values: list[float] = []
        for f in range(len(vars)):
            elems += [vars[indices[f]]]
            imp_values.append(importances[indices[f]])
            print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

        figure(figsize=(11, 11))
        plot_horizontal_bar_chart(
            elems,
            imp_values,
            error=stdevs,
            title="RF variables importance",
            xlabel="importance",
            ylabel="variables",
            percentage=True,
        )
        savefig(f"graphs/classification/data_modeling/rf/{self.data_loader.file_tag}_rf_{eval_metric}_vars_ranking.png")

        # Overfitting Study
        d_max: int = params["params"][0]
        feat: float = params["params"][1]
        nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

        y_tst_values: list[float] = []
        y_trn_values: list[float] = []
        acc_metric: str = "accuracy"

        for n in nr_estimators:
            clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat)
            clf.fit(trnX, trnY)
            prd_tst_Y: array = clf.predict(tstX)
            prd_trn_Y: array = clf.predict(trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            nr_estimators,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"RF overfitting study for d={d_max} and f={feat}",
            xlabel="nr_estimators",
            ylabel=str(eval_metric),
            percentage=True,
        )
        savefig(f"graphs/classification/data_modeling/rf/{self.data_loader.file_tag}_rf_{eval_metric}_overfitting.png")

        return evaluation_results

    def gradient_boosting_study(self,
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array,
        nr_max_trees: int = 2500, lag: int = 500, metric: str = "accuracy",
    ) -> tuple[GradientBoostingClassifier | None, dict]:
        n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
        max_depths: list[int] = [2, 5, 7]
        learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

        best_model: GradientBoostingClassifier | None = None
        best_params: dict = {"name": "GB", "metric": metric, "params": ()}
        best_performance: float = 0.0

        values: dict = {}
        cols: int = len(max_depths)
        _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
        for i in range(len(max_depths)):
            d: int = max_depths[i]
            values = {}
            for lr in learning_rates:
                y_tst_values: list[float] = []
                for n in n_estimators:
                    clf = GradientBoostingClassifier(
                        n_estimators=n, max_depth=d, learning_rate=lr
                    )
                    clf.fit(trnX, trnY)
                    prdY: array = clf.predict(tstX)
                    eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                    y_tst_values.append(eval)
                    if eval - best_performance > DELTA_IMPROVE:
                        best_performance = eval
                        best_params["params"] = (d, lr, n)
                        best_params[metric] = eval
                        best_model = clf
                    # print(f'GB d={d} lr={lr} n={n}')
                values[lr] = y_tst_values
            plot_multiline_chart(
                n_estimators,
                values,
                ax=axs[0, i],
                title=f"Gradient Boosting with max_depth={d}",
                xlabel="nr estimators",
                ylabel=metric,
                percentage=True,
            )
        print(f'GB best for {best_params["params"][2]} trees (d={best_params["params"][0]} and lr={best_params["params"][1]}')

        return best_model, best_params

    def gradient_boosting_classification(self):
        eval_metric = "recall"

        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values
        labels: list = list(self.y_train.unique())
        labels.sort()
        vars = self.X_train.columns.to_list()
        
        print(f"Train#={len(trnX)} Test#={len(tstX)}")
        print(f"Labels={labels}")

        # Recall Study
        figure()
        best_model, params = self.gradient_boosting_study(
            trnX,
            trnY,
            tstX,
            tstY,
            nr_max_trees=1000,
            lag=250,
            metric=eval_metric,
        )
        savefig(f"graphs/classification/data_modeling/gb/{self.data_loader.file_tag}_gb_{eval_metric}_study.png")
        show()

        print(f"Best Model: {params['name']} | Metric: {params['metric']} | Score: {params[params['metric']]}")

        # Best Model Performance Analysis
        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)
        figure()
        evaluation_results = self.plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(f'graphs/classification/data_modeling/gb/{self.data_loader.file_tag}_gb_{params["name"]}_best_{params["metric"]}_eval.png')
        show()

        # Variables Importance Study
        trees_importances: list[float] = []
        for lst_trees in best_model.estimators_:
            for tree in lst_trees:
                trees_importances.append(tree.feature_importances_)

        stdevs: list[float] = list(std(trees_importances, axis=0))
        importances = best_model.feature_importances_
        indices: list[int] = argsort(importances)[::-1]
        elems: list[str] = []
        imp_values: list[float] = []
        for f in range(len(vars)):
            elems += [vars[indices[f]]]
            imp_values.append(importances[indices[f]])
            print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

        figure(figsize=(11, 11))
        plot_horizontal_bar_chart(
            elems,
            imp_values,
            error=stdevs,
            title="GB variables importance",
            xlabel="importance",
            ylabel="variables",
            percentage=True,
        )
        savefig(f"graphs/classification/data_modeling/gb/{self.data_loader.file_tag}_gb_{eval_metric}_vars_ranking.png")

        # Overfitting Study
        d_max: int = params["params"][0]
        lr: float = params["params"][1]
        nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

        y_tst_values: list[float] = []
        y_trn_values: list[float] = []
        acc_metric: str = "accuracy"

        for n in nr_estimators:
            clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr)
            clf.fit(trnX, trnY)
            prd_tst_Y: array = clf.predict(tstX)
            prd_trn_Y: array = clf.predict(trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            nr_estimators,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"GB overfitting study for d={d_max} and lr={lr}",
            xlabel="nr_estimators",
            ylabel=str(eval_metric),
            percentage=True,
        )
        savefig(f"graphs/classification/data_modeling/gb/{self.data_loader.file_tag}_gb_{eval_metric}_overfitting.png")

        return evaluation_results

    # %% Forecasting

    class SimpleAvgRegressor(RegressorMixin):
        def __init__(self):
            super().__init__()
            self.mean: float = 0.0
            return

        def fit(self, X: Series):
            self.mean = X.mean()
            return

        def predict(self, X: Series) -> Series:
            prd: list = len(X) * [self.mean]
            prd_series: Series = Series(prd)
            prd_series.index = X.index
            return prd_series

    def simple_average_model_forecasting(self):

        fr_mod = self.SimpleAvgRegressor()
        fr_mod.fit(self.y_train)

        prd_trn: Series = fr_mod.predict(self.y_train)
        prd_tst: Series = fr_mod.predict(self.y_test)

        plot_forecasting_eval(self.y_train, self.y_test, prd_trn, prd_tst, title=f"{self.data_loader.file_tag} - Simple Average")
        savefig(f"graphs/forecasting/data_modeling/simple_average/{self.data_loader.file_tag}_simpleAvg_eval.png")

        plot_forecasting_series(
            self.y_train,
            self.y_test,
            prd_tst,
            title=f"{self.data_loader.file_tag} - Simple Average",
            xlabel=self.data_loader.read_options["index_col"],
            ylabel=self.data_loader.target,
        )
        savefig(f"graphs/forecasting/data_modeling/simple_average/{self.data_loader.file_tag}_simpleAvg_forecast.png")
        show()

        return {
            "R2_train": FORECAST_MEASURES["R2"](self.y_train, prd_trn),
            "R2_test": FORECAST_MEASURES["R2"](self.y_test, prd_tst),
        }

    class PersistenceOptimistRegressor(RegressorMixin):
        def __init__(self):
            super().__init__()
            self.last: float = 0.0
            return

        def fit(self, X: Series):
            self.last = X.iloc[-1]
            # print(self.last)
            return

        def predict(self, X: Series):
            prd: list = X.shift().values.ravel()
            prd[0] = self.last
            prd_series: Series = Series(prd)
            prd_series.index = X.index
            return prd_series

    class PersistenceRealistRegressor(RegressorMixin):
        def __init__(self):
            super().__init__()
            self.last = 0
            self.estimations = [0]
            self.obs_len = 0

        def fit(self, X: Series):
            for i in range(1, len(X)):
                self.estimations.append(X.iloc[i - 1])
            self.obs_len = len(self.estimations)
            self.last = X.iloc[len(X) - 1]
            prd_series: Series = Series(self.estimations)
            prd_series.index = X.index
            return prd_series

        def predict(self, X: Series):
            prd: list = len(X) * [self.last]
            prd_series: Series = Series(prd)
            prd_series.index = X.index
            return prd_series

    def persistence_model_forecasting(self):

        eval_results = {}

        # Persistence Forecasting Optimist
        fr_mod = self.PersistenceOptimistRegressor()
        fr_mod.fit(self.y_train)

        prd_trn: Series = fr_mod.predict(self.y_train)
        prd_tst: Series = fr_mod.predict(self.y_test)

        plot_forecasting_eval(self.y_train, self.y_test, prd_trn, prd_tst, title=f"{self.data_loader.file_tag} - Persistence Optimist")
        savefig(f"graphs/forecasting/data_modeling/persistence/{self.data_loader.file_tag}_persistence_optim_eval.png")

        plot_forecasting_series(
            self.y_train,
            self.y_test,
            prd_tst,
            title=f"{self.data_loader.file_tag} - Persistence Optimist",
            xlabel=self.data_loader.read_options["index_col"],
            ylabel=self.data_loader.target,
        )
        savefig(f"graphs/forecasting/data_modeling/persistence/{self.data_loader.file_tag}_persistence_optim_forecast.png")
        show()

        eval_results["Persistence Optimist"] = {
            "R2_train": FORECAST_MEASURES["R2"](self.y_train, prd_trn),
            "R2_test": FORECAST_MEASURES["R2"](self.y_test, prd_tst),
        }

        # Persistence Forecasting Realist
        fr_mod = self.PersistenceRealistRegressor()
        fr_mod.fit(self.y_train)

        prd_trn: Series = fr_mod.predict(self.y_train)
        prd_tst: Series = fr_mod.predict(self.y_test)

        plot_forecasting_eval(self.y_train, self.y_test, prd_trn, prd_tst, title=f"{self.data_loader.file_tag} - Persistence Realist")
        savefig(f"graphs/forecasting/data_modeling/persistence/{self.data_loader.file_tag}_persistence_real_eval.png")

        plot_forecasting_series(
            self.y_train,
            self.y_test,
            prd_tst,
            title=f"{self.data_loader.file_tag} - Persistence Realist",
            xlabel=self.data_loader.read_options["index_col"],
            ylabel=self.data_loader.target,
        )
        savefig(f"graphs/forecasting/data_modeling/persistence/{self.data_loader.file_tag}_persistence_real_forecast.png")
        show()

        eval_results["Persistence Realist"] = {
            "R2_train": FORECAST_MEASURES["R2"](self.y_train, prd_trn),
            "R2_test": FORECAST_MEASURES["R2"](self.y_test, prd_tst),
        }

        return eval_results

    class RollingMeanRegressor(RegressorMixin):
        def __init__(self, win: int = 3):
            super().__init__()
            self.win_size = win
            self.memory: list = []

        def fit(self, X: Series):
            """
            Store the last `win_size` values of the training series for predictions.
            """
            self.memory = X.iloc[-self.win_size:].tolist()
            return self

        def predict(self, X: Series):
            """
            Generate predictions using a rolling mean approach.
            """
            if len(self.memory) < self.win_size:
                raise ValueError("Not enough data in memory to compute rolling mean.")

            # Initialize estimations with the memory values
            estimations = self.memory.copy()

            # Generate predictions for the length of X
            predictions = []
            for _ in range(len(X)):
                # Compute the rolling mean and add it to estimations
                new_value = mean(estimations[-self.win_size:])
                predictions.append(new_value)
                estimations.append(new_value)

            # Create a Series with the predictions, aligned with X's index
            prd_series = Series(predictions, index=X.index)
            return prd_series

    def rolling_mean_study(self, train: Series, test: Series, measure: str = "R2"):
        """
        Performs a study to evaluate different window sizes for a rolling mean model.

        Args:
            train (Series): Training dataset.
            test (Series): Testing dataset.
            measure (str): Evaluation metric, either "R2" or "MAPE".

        Returns:
            best_model: The best RollingMeanRegressor model based on the evaluation metric.
            best_params (dict): Information about the best-performing parameters.
        """
        # Define possible window sizes
        win_size = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

        # Check if the evaluation metric is supported
        if measure not in FORECAST_MEASURES:
            raise ValueError(f"Unsupported measure: {measure}. Choose from {list(FORECAST_MEASURES.keys())}.")

        flag = measure == "R2" or measure == "MAPE"
        best_model = None
        best_params: dict = {"name": "Rolling Mean", "metric": measure, "params": ()}
        best_performance: float = -float('inf')  # Initialize for maximization (e.g., R2)

        yvalues = []

        # Iterate over each window size
        for w in win_size:
            # Ensure the training data is large enough for the current window size
            if len(train) < w:
                print(f"Skipping window size {w} as train size ({len(train)}) is smaller than the window.")
                yvalues.append(None)  # No evaluation for this window size
                continue

            try:
                # Initialize and fit the rolling mean model
                pred = self.RollingMeanRegressor(win=w)
                pred.fit(train)

                # Ensure enough data for prediction
                if len(test) < w:
                    print(f"Skipping window size {w} as test size ({len(test)}) is smaller than the window.")
                    yvalues.append(None)
                    continue

                # Predict and evaluate
                prd_tst = pred.predict(test)
                eval: float = FORECAST_MEASURES[measure](test, prd_tst)

                # Track the best-performing model
                if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (w,)
                    best_model = pred

                yvalues.append(eval)
            except Exception as e:
                print(f"Error with window size {w}: {e}")
                yvalues.append(None)

        # Display the best result
        if best_model is not None:
            print(f"Rolling Mean best with win={best_params['params'][0]} -> {measure}={best_performance:.4f}")
        else:
            print("No suitable window size found for Rolling Mean.")

        # Plot performance across window sizes
        valid_yvalues = [v for v in yvalues if v is not None]
        valid_win_sizes = [w for w, v in zip(win_size, yvalues) if v is not None]

        if valid_yvalues:
            plot_line_chart(
                valid_win_sizes, valid_yvalues,
                title=f"Rolling Mean ({measure})",
                xlabel="Window Size",
                ylabel=measure,
                percentage=flag,
            )
        else:
            print("No valid results to plot.")

        return best_model, best_params

    def rolling_mean_model_forecasting(self):

        measure: str = "R2"

        fig = figure(figsize=(HEIGHT, HEIGHT))
        best_model, best_params = self.rolling_mean_study(self.y_train, self.y_test)
        savefig(f"graphs/forecasting/data_modeling/rolling_mean/{self.data_loader.file_tag}_rollingmean_{measure}_study.png")
        show()

        params = best_params["params"]
        prd_trn: Series = best_model.predict(self.y_train)
        prd_tst: Series = best_model.predict(self.y_test)

        plot_forecasting_eval(self.y_train, self.y_test, prd_trn, prd_tst, title=f"{self.data_loader.file_tag} - Rolling Mean (win={params[0]})")
        savefig(f"graphs/forecasting/data_modeling/rolling_mean/{self.data_loader.file_tag}_rollingmean_{measure}_win{params[0]}_eval.png")
        show()

        plot_forecasting_series(
            self.y_train,
            self.y_test,
            prd_tst,
            title=f"{self.data_loader.file_tag} - Rolling Mean (win={params[0]})",
            xlabel=self.data_loader.read_options["index_col"],
            ylabel=self.data_loader.target,
        )
        savefig(f"graphs/forecasting/data_modeling/rolling_mean/{self.data_loader.file_tag}_rollingmean_{measure}_forecast.png")
        show()

        return {
            "R2_train": FORECAST_MEASURES["R2"](self.y_train, prd_trn),
            "R2_test": FORECAST_MEASURES["R2"](self.y_test, prd_tst),
        }

    def exponential_smoothing_study(self, train: Series, test: Series, measure: str = "R2"):
        alpha_values = [i / 10 for i in range(1, 10)]
        flag = measure == "R2" or measure == "MAPE"
        best_model = None
        best_params: dict = {"name": "Exponential Smoothing", "metric": measure, "params": ()}
        best_performance: float = -100000

        yvalues = []
        for alpha in alpha_values:
            tool = SimpleExpSmoothing(train)
            model = tool.fit(smoothing_level=alpha, optimized=False)
            prd_tst = model.forecast(steps=len(test))

            eval: float = FORECAST_MEASURES[measure](test, prd_tst)
            # print(w, eval)
            if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                best_performance: float = eval
                best_params["params"] = (alpha,)
                best_model = model
            yvalues.append(eval)

        print(f"Exponential Smoothing best with alpha={best_params['params'][0]:.0f} -> {measure}={best_performance}")
        plot_line_chart(
            alpha_values,
            yvalues,
            title=f"Exponential Smoothing ({measure})",
            xlabel="alpha",
            ylabel=measure,
            percentage=flag,
        )

        return best_model, best_params

    def exponential_smoothing_model_forecasting(self):

        measure: str = "R2"

        best_model, best_params = self.exponential_smoothing_study(self.y_train, self.y_test, measure=measure)
        savefig(f"graphs/forecasting/data_modeling/exponential_smoothing/{self.data_loader.file_tag}_exponential_smoothing_{measure}_study.png")
        show()

        params = best_params["params"]
        prd_trn = best_model.predict(start=0, end=len(self.y_train) - 1)
        prd_tst = best_model.forecast(steps=len(self.y_test))

        plot_forecasting_eval(self.y_train, self.y_test, prd_trn, prd_tst,
                              title=f"{self.data_loader.file_tag} - Exponential Smoothing alpha={params[0]}")
        savefig(f"graphs/forecasting/data_modeling/exponential_smoothing/{self.data_loader.file_tag}_exponential_smoothing_{measure}_eval.png")
        show()

        plot_forecasting_series(
            self.y_train,
            self.y_test,
            prd_tst,
            title=f"{self.data_loader.file_tag} - Exponential Smoothing ",
            xlabel=self.data_loader.read_options["index_col"],
            ylabel=self.data_loader.target,
        )
        savefig(f"graphs/forecasting/data_modeling/exponential_smoothing/{self.data_loader.file_tag}_exponential_smoothing_{measure}_forecast.png")
        show()

        return {
            "R2_train": FORECAST_MEASURES["R2"](self.y_train, prd_trn),
            "R2_test": FORECAST_MEASURES["R2"](self.y_test, prd_tst),
        }

    def linear_regression_model_forecasting(self):

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        prd_trn: Series = Series(model.predict(self.X_train), index=self.y_train.index)
        prd_tst: Series = Series(model.predict(self.X_test), index=self.y_test.index)

        plot_forecasting_eval(self.y_train, self.y_test, prd_trn, prd_tst, title=f"{self.data_loader.file_tag} - Linear Regression")
        savefig(f"graphs/forecasting/data_modeling/lr/{self.data_loader.file_tag}_linear_regression_eval.png")
        show()

        plot_forecasting_series(
            self.y_train,
            self.y_test,
            prd_tst,
            title=f"{self.data_loader.file_tag} - Linear Regression",
            xlabel=self.data_loader.read_options["index_col"],
            ylabel=self.data_loader.target,
        )
        savefig(f"graphs/forecasting/data_modeling/lr/{self.data_loader.file_tag}_linear_regression_forecast.png")
        show()

        return {
            "R2_train": FORECAST_MEASURES["R2"](self.y_train, prd_trn),
            "R2_test": FORECAST_MEASURES["R2"](self.y_test, prd_tst),
        }

    def arima_study(self, train: Series, test: Series, measure: str = "R2"):
        d_values = (0, 1, 2)
        p_params = (1, 2, 3, 5, 7, 10)
        q_params = (1, 3, 5, 7)

        flag = measure == "R2" or measure == "MAPE"
        best_model = None
        best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
        best_performance: float = -100000

        fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
        for i in range(len(d_values)):
            d: int = d_values[i]
            values = {}
            for q in q_params:
                yvalues = []
                for p in p_params:
                    arima = ARIMA(train, order=(p, d, q))
                    model = arima.fit()
                    prd_tst = model.forecast(steps=len(test), signal_only=False)
                    eval: float = FORECAST_MEASURES[measure](test, prd_tst)
                    # print(f"ARIMA ({p}, {d}, {q})", eval)
                    if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                        best_performance: float = eval
                        best_params["params"] = (p, d, q)
                        best_model = model
                    yvalues.append(eval)
                values[q] = yvalues
            plot_multiline_chart(
                p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure,
                percentage=flag
            )
        print(
            f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
        )

        return best_model, best_params

    def arima_model_forecasting(self, multivariate: bool = False):

        measure: str = "R2"

        predictor = ARIMA(self.y_train, order=(3, 1, 2))
        model = predictor.fit()
        print(model.summary())

        model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))
        show()

        best_model, best_params = self.arima_study(self.y_train, self.y_test, measure=measure)
        if multivariate:
            savefig(f"graphs/forecasting/data_modeling/arima/{self.data_loader.file_tag}_arima_{measure}_study_multivariate.png")
        else:
            savefig(f"graphs/forecasting/data_modeling/arima/{self.data_loader.file_tag}_arima_{measure}_study.png")
        show()

        params = best_params["params"]
        prd_trn = best_model.predict(start=0, end=len(self.y_train) - 1)
        prd_tst = best_model.forecast(steps=len(self.y_test))

        plot_forecasting_eval(
            self.y_train, self.y_test, prd_trn, prd_tst, title=f"{self.data_loader.file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})"
        )
        if multivariate:
            savefig(f"graphs/forecasting/data_modeling/arima/{self.data_loader.file_tag}_arima_{measure}_eval_multivariate.png")
        else:
            savefig(f"graphs/forecasting/data_modeling/arima/{self.data_loader.file_tag}_arima_{measure}_eval.png")
        show()

        plot_forecasting_series(
            self.y_train,
            self.y_test,
            prd_tst,
            title=f"{self.data_loader.file_tag} - ARIMA ",
            xlabel=self.data_loader.read_options["index_col"],
            ylabel=self.data_loader.target,
        )
        if multivariate:
            savefig(f"graphs/forecasting/data_modeling/arima/{self.data_loader.file_tag}_arima_{measure}_forecast_multivariate.png")
        else:
            savefig(f"graphs/forecasting/data_modeling/arima/{self.data_loader.file_tag}_arima_{measure}_forecast.png")
        show()

        return {
            "R2_train": FORECAST_MEASURES["R2"](self.y_train, prd_trn),
            "R2_test": FORECAST_MEASURES["R2"](self.y_test, prd_tst),
        }

    def prepare_dataset_for_lstm(self, series, seq_length: int = 4):
        # Adjust sequence length if it exceeds the series size
        seq_length = min(seq_length, len(series) - 1)
        setX: list = []
        setY: list = []
        for i in range(len(series) - seq_length):
            past = series[i: i + seq_length]
            future = series[i + seq_length]
            setX.append(past)
            setY.append(future)
        return tensor(setX).float().unsqueeze(-1), tensor(setY).float().unsqueeze(-1)

    class DS_LSTM(Module):

        def __init__(self, train, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1, length: int = 4):
            super().__init__()
            self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.linear = Linear(hidden_size, 1)
            self.optimizer = Adam(self.parameters())
            self.loss_fn = MSELoss()

            trnX, trnY = self.prepare_dataset_for_lstm(train, seq_length=length)
            self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=max(1, len(trnX) // 10))

        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.linear(x[:, -1, :])  # Use only the last time step's output
            return x

        def fit(self):
            self.train()
            for batchX, batchY in self.loader:
                y_pred = self(batchX)
                loss = self.loss_fn(y_pred, batchY)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss.item()

        def predict(self, X):
            with no_grad():
                y_pred = self(X)
            return y_pred

        def prepare_dataset_for_lstm(self, series, seq_length: int = 4):
            # Adjust sequence length if it exceeds the series size
            seq_length = min(seq_length, len(series) - 1)
            setX: list = []
            setY: list = []
            for i in range(len(series) - seq_length):
                past = series[i: i + seq_length]
                future = series[i + seq_length]
                setX.append(past)
                setY.append(future)
            return tensor(setX).float().unsqueeze(-1), tensor(setY).float().unsqueeze(-1)

    def lstm_study(self, train, test, nr_episodes: int = 1000, measure: str = "R2"):
        sequence_size = [2, 4, 8]
        nr_hidden_units = [25, 50, 100]

        step: int = nr_episodes // 10
        episodes = [1] + list(range(0, nr_episodes + 1, step))[1:]
        flag = measure in ["R2", "MAPE"]
        best_model = None
        best_params: dict = {"name": "LSTM", "metric": measure, "params": ()}
        best_performance: float = -float("inf")
        delta_improve = 1e-6

        # Filter sequence_size to ensure valid lengths based on the test set
        valid_sequence_size = [length for length in sequence_size if length <= len(test) - 1]

        # Create subplots dynamically
        _, axs = subplots(1, len(valid_sequence_size), figsize=(len(valid_sequence_size) * HEIGHT, HEIGHT))
        if len(valid_sequence_size) == 1:
            axs = [axs]  # Wrap single Axes object in a list

        for i, length in enumerate(valid_sequence_size):
            tstX, tstY = self.prepare_dataset_for_lstm(test, seq_length=length)

            values = {}
            for hidden in nr_hidden_units:
                yvalues = []
                model = self.DS_LSTM(train, input_size=1, hidden_size=hidden, length=length)
                for n in range(0, nr_episodes + 1):
                    model.fit()
                    if n % step == 0:
                        prd_tst = model.predict(tstX)
                        eval: float = FORECAST_MEASURES[measure](test[length:], prd_tst)
                        print(f"seq length={length} hidden_units={hidden} nr_episodes={n} ==> {measure}: {eval:.2f}")
                        if eval > best_performance and abs(eval - best_performance) > delta_improve:
                            best_performance = eval
                            best_params["params"] = (length, hidden, n)
                            best_model = deepcopy(model)
                        yvalues.append(eval)
                values[hidden] = yvalues

            # Plot on the appropriate subplot
            plot_multiline_chart(
                episodes,
                values,
                ax=axs[i],
                title=f"LSTM seq length={length} ({measure})",
                xlabel="nr episodes",
                ylabel=measure,
                percentage=flag,
            )

        print(
            f"LSTM best results achieved with seq length={best_params['params'][0]}, "
            f"hidden_units={best_params['params'][1]}, and nr_episodes={best_params['params'][2]} ==> "
            f"{measure}={best_performance:.2f}"
        )
        return best_model, best_params

    def lstm_model_forecasting(self, multivariate: bool = False):
        measure: str = "R2"

        # Convert Y_train and Y_test to numpy arrays, keeping the original index
        Y_train = self.y_train.values.astype("float32")
        Y_test = self.y_test.values.astype("float32")

        print(f"Train shape={Y_train.shape} Test shape={Y_test.shape}")

        # Initialize and train the LSTM model
        model = self.DS_LSTM(Y_train, input_size=1, hidden_size=50, num_layers=1)
        loss = model.fit()
        print(f"Training Loss: {loss}")

        # Perform LSTM study to find the best model and parameters
        best_model, best_params = self.lstm_study(Y_train, Y_test, nr_episodes=3000, measure=measure)
        if multivariate:
            savefig(f"graphs/forecasting/data_modeling/lstm/{self.data_loader.file_tag}_lstms_{measure}_study_multivariate.png")
        else:
            savefig(f"graphs/forecasting/data_modeling/lstm/{self.data_loader.file_tag}_lstms_{measure}_study.png")
        show()

        # Extract best parameters
        params = best_params["params"]
        best_length = params[0]

        # Prepare datasets with the best sequence length
        trnX, trnY = self.prepare_dataset_for_lstm(Y_train, seq_length=best_length)
        tstX, tstY = self.prepare_dataset_for_lstm(Y_test, seq_length=best_length)

        # Make predictions using the best model
        prd_trn = best_model.predict(trnX)
        prd_tst = best_model.predict(tstX)

        # Convert predictions and data back to pandas Series for plotting
        prd_trn_series = Series(prd_trn.numpy().ravel(), index=self.y_train.index[best_length:])
        prd_tst_series = Series(prd_tst.numpy().ravel(), index=self.y_test.index[best_length:])

        # Plot evaluation of forecasting
        plot_forecasting_eval(
            self.y_train[best_length:],  # Convert to Series
            self.y_test[best_length:],  # Convert to Series
            prd_trn_series,
            prd_tst_series,
            title=f"{self.data_loader.file_tag} - LSTM (length={best_length}, hidden={params[1]}, epochs={params[2]})",
        )
        if multivariate:
            savefig(f"graphs/forecasting/data_modeling/lstm/{self.data_loader.file_tag}_lstms_{measure}_eval_multivariate.png")
        else:
            savefig(f"graphs/forecasting/data_modeling/lstm/{self.data_loader.file_tag}_lstms_{measure}_eval.png")
        show()

        # Create a pandas Series for the predicted test series
        pred_series: Series = Series(prd_tst.numpy().ravel(), index=self.y_test.index[best_length:])

        # Plot forecasting series
        plot_forecasting_series(
            self.y_train[best_length:],
            self.y_test[best_length:],
            pred_series,
            title=f"{self.data_loader.file_tag} - LSTMs ",
            xlabel=self.data_loader.read_options["index_col"],
            ylabel=self.data_loader.target,
        )
        if multivariate:
            savefig(f"graphs/forecasting/data_modeling/lstm/{self.data_loader.file_tag}_lstms_{measure}_forecast_multivariate.png")
        else:
            savefig(f"graphs/forecasting/data_modeling/lstm/{self.data_loader.file_tag}_lstms_{measure}_forecast.png")
        show()

        return {
            "R2_train": FORECAST_MEASURES["R2"](self.y_train[best_length:], prd_trn_series),
            "R2_test": FORECAST_MEASURES["R2"](self.y_test[best_length:], prd_tst_series),
        }