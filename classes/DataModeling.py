from matplotlib.figure import Figure
from numpy import array, ndarray, argsort, arange
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show, title, imshow, imread, axis, subplots
from typing import Literal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from subprocess import call

from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, \
    plot_multiline_chart, plot_horizontal_bar_chart, HEIGHT, plot_line_chart, plot_multibar_chart, plot_confusion_matrix


class DataModeling:

    def __init__(self, data_loader, X_train, X_test, y_train, y_test):
        self.data_loader = data_loader
        self.target = self.data_loader.target
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.squeeze()
        self.y_test = y_test.squeeze()

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

        return best_model, best_params, best_performance

    def plot_evaluation_results(self, model, trn_y, prd_trn, tst_y, prd_tst, labels: ndarray, dataset) -> ndarray:
        evaluation: dict = {}
        for key in CLASS_EVAL_METRICS:
            evaluation[key] = [
                CLASS_EVAL_METRICS[key](trn_y, prd_trn),
                CLASS_EVAL_METRICS[key](tst_y, prd_tst),
            ]

        params_st: str = "" if () == model["params"] else str(model["params"])
        fig: Figure
        axs: ndarray
        fig, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
        fig.suptitle(f'Best {model["metric"]} for {model["name"]} {params_st} - {dataset}')
        plot_multibar_chart(["Train", "Test"], evaluation, ax=axs[0], percentage=True)

        cnf_mtx_tst: ndarray = confusion_matrix(tst_y, prd_tst, labels=labels)
        plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1])
        return axs

    def evaluate_naive_bayes(self):
        """
        Evaluate Naive Bayes models for both accuracy and recall and generate plots for each metric.
        Returns the results for both evaluations.
        """
        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values

        print(f"Train#={len(trnX)} Test#={len(tstX)}")

        # Accuracy
        figure(figsize=(7, 5))
        best_model_acc, params_acc, best_acc = self.naive_Bayes_study(trnX, trnY, tstX, tstY, metric="accuracy")
        savefig(f"graphs/data_modeling/nb/{self.data_loader.file_tag}_nb_accuracy_study.png")
        show()

        # Recall
        figure(figsize=(7, 5))
        best_model_recall, params_recall, best_recall = self.naive_Bayes_study(trnX, trnY, tstX, tstY, metric="recall")
        savefig(f"graphs/data_modeling/nb/{self.data_loader.file_tag}_nb_recall_study.png")
        show()

        # Print model performance
        print(f"\nModel Performance:")
        print(f"Accuracy - Best Model: {params_acc['name']} | Accuracy: {best_acc}")
        print(f"Recall - Best Model: {params_recall['name']} | Recall: {best_recall}")

        # Return models and their parameters
        return {
            params_acc["name"]: {"metric": "accuracy", "model": best_model_acc, "params": params_acc,
                                 "score": best_acc},
            params_recall["name"]: {"metric": "recall", "model": best_model_recall, "params": params_recall,
                                    "score": best_recall},
        }, trnX, trnY, tstX, tstY

    def analyze_naive_bayes(self, chosen_model, chosen_params, trnX, trnY, tstX, tstY):
        """
        Perform performance analysis for the chosen Naive Bayes model.

        Parameters:
            chosen_model: The model selected for analysis.
            chosen_params: The parameters of the selected model.
        """
        labels: list = list(self.y_train.unique())
        labels.sort()

        prd_trn: array = chosen_model.predict(trnX)
        prd_tst: array = chosen_model.predict(tstX)

        # Plot evaluation results
        figure()
        self.plot_evaluation_results(chosen_params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(
            f'graphs/data_modeling/nb/{self.data_loader.file_tag}_{chosen_params["name"]}_best_{chosen_params["metric"]}_eval.png')
        show()

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
                    best_model = clf
                # print(f'KNN {d} k={k}')
            values[d] = y_tst_values
        print(f"KNN best with k={best_params['params'][0]} and {best_params['params'][1]}")
        plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric}) - {dataset}', xlabel='k', ylabel=metric,
                             percentage=True)

        return best_model, best_params

    def evaluate_knn(self, k_max: int = 25, lag: int = 2) -> dict:
        """
        Evaluate KNN models for both accuracy and recall and generate plots for each metric.
        Returns the results for both evaluations.
        """
        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values

        print(f"Train#={len(trnX)} Test#={len(tstX)}")

        # Accuracy Study
        figure()
        best_model_acc, params_acc = self.knn_study(
            trnX, trnY, tstX, tstY, self.data_loader.file_tag, k_max=k_max, lag=lag, metric="accuracy"
        )
        savefig(f"graphs/data_modeling/knn/{self.data_loader.file_tag}_knn_accuracy_study.png")
        show()

        # Recall Study
        figure()
        best_model_recall, params_recall = self.knn_study(
            trnX, trnY, tstX, tstY, self.data_loader.file_tag, k_max=k_max, lag=lag, metric="recall"
        )
        savefig(f"graphs/data_modeling/knn/{self.data_loader.file_tag}_knn_recall_study.png")
        show()

        # Print model performance
        print("\nModel Performance:")
        print(f"Accuracy - Best Params: {params_acc}")
        print(f"Recall - Best Params: {params_recall}")

        # Return models and their parameters
        return {
            params_acc['params'][1]: {"metric": "accuracy", "model": best_model_acc, "params": params_acc},
            params_recall['params'][1]: {"metric": "recall", "model": best_model_recall, "params": params_recall},
        }, trnX, trnY, tstX, tstY

    def analyze_knn(self, chosen_model: KNeighborsClassifier, chosen_params: dict, metric, trnX, trnY, tstX, tstY):
        """
        Perform performance analysis and overfitting study for the chosen KNN model.

        Parameters:
            chosen_model: The model selected for analysis.
            chosen_params: The parameters of the selected model.
            metric: Evaluation metric used for overfitting study (default: recall).
        """
        labels: list = list(self.y_train.unique())
        labels.sort()

        # Best Model Performance Analysis
        prd_trn: array = chosen_model.predict(trnX)
        prd_tst: array = chosen_model.predict(tstX)
        figure()
        self.plot_evaluation_results(chosen_params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(
            f'graphs/data_modeling/knn/{self.data_loader.file_tag}_knn_{chosen_params["name"]}_best_{metric}_eval.png'
        )
        show()

        # Overfitting Study
        distance = chosen_params["params"][1]
        K_MAX = 25
        kvalues = [i for i in range(1, K_MAX, 2)]
        y_tst_values = []
        y_trn_values = []

        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
            clf.fit(trnX, trnY)
            prd_tst_Y: array = clf.predict(tstX)
            prd_trn_Y: array = clf.predict(trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[metric](tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[metric](trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            kvalues,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"KNN Overfitting Study for {distance} - {self.data_loader.file_tag}",
            xlabel="K",
            ylabel=metric,
            percentage=True,
        )
        savefig(f"graphs/data_modeling/knn/{self.data_loader.file_tag}_knn_overfitting.png")
        show()

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
                    best_model = clf
                # print(f'DT {c} and d={d}')
            values[c] = y_tst_values
        print(f"DT best with {best_params['params'][0]} and d={best_params['params'][1]}")
        plot_multiline_chart(depths, values, title=f'DT Models ({metric}) - {dataset}', xlabel='d', ylabel=metric, percentage=True)

        return best_model, best_params

    def evaluate_decision_tree(self):

        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values

        print(f'Train#={len(trnX)} Test#={len(tstX)}')

        # Accuracy Study
        figure()
        best_model_acc, params_acc = self.trees_study(trnX, trnY, tstX, tstY, self.data_loader.file_tag, d_max=25,
                                              metric='accuracy')
        savefig(f'graphs/data_modeling/dt/{self.data_loader.file_tag}_dt_accuracy_study.png')
        show()

        # Recall Study
        figure()
        best_model_recall, params_recall = self.trees_study(trnX, trnY, tstX, tstY, self.data_loader.file_tag, d_max=25,
                                              metric='recall')
        savefig(f'graphs/data_modeling/dt/{self.data_loader.file_tag}_dt_recall_study.png')
        show()

        # Print model performance
        print("\nModel Performance:")
        print(f"Accuracy - Best Params: {params_acc}")
        print(f"Recall - Best Params: {params_recall}")

        # Return models and their parameters
        return {
            params_acc['params'][0]: {"metric": "accuracy", "model": best_model_acc, "params": params_acc},
            params_recall['params'][0]: {"metric": "recall", "model": best_model_recall, "params": params_recall},
        }, trnX, trnY, tstX, tstY

    def analyze_decision_tree(self, chosen_model, chosen_params, metric, trnX, trnY, tstX, tstY):

        labels: list = list(self.y_train.unique())
        labels.sort()
        vars = self.X_train.columns.to_list()

        # Best Model Performance Analysis
        prd_trn: array = chosen_model.predict(trnX)
        prd_tst: array = chosen_model.predict(tstX)
        figure()
        self.plot_evaluation_results(chosen_params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(
            f'graphs/data_modeling/dt/{self.data_loader.file_tag}_dt_{chosen_params["name"]}_best_{chosen_params["metric"]}_eval.png')
        show()

        # Variables Importance
        tree_filename: str = f"graphs/data_modeling/dt/{self.data_loader.file_tag}_dt_{metric}_best_tree"
        max_depth2show = 3
        st_labels: list[str] = [str(value) for value in labels]

        figure(figsize=(14, 6))
        plot_tree(
            chosen_model,
            max_depth=max_depth2show,
            feature_names=vars,
            class_names=st_labels,
            filled=True,
            rounded=True,
            impurity=False,
            precision=2,
        )
        savefig(tree_filename + ".png")

        importances = chosen_model.feature_importances_
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
        savefig(f"graphs/data_modeling/dt/{self.data_loader.file_tag}_dt_{metric}_vars_ranking.png")

        # Overfitting Study
        crit: Literal["entropy", "gini"] = chosen_params["params"][0]
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
            ylabel=str(metric),
            percentage=True,
        )
        savefig(f"graphs/data_modeling/dt/{self.data_loader.file_tag}_dt_{metric}_overfitting.png")
        show()

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

    def mlp(self):

        LAG: int = 500
        NR_MAX_ITER: int = 5000
        eval_metric = "accuracy"

        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values
        labels: list = list(self.y_train.unique())
        labels.sort()
        vars = self.X_train.columns.to_list()

        print(f'Train#={len(trnX)} Test#={len(tstX)}')
        print(f'Labels={labels}')

        # Accuracy Study
        figure()
        best_model, params = self.mlp_study(trnX, trnY, tstX, tstY, nr_max_iterations=NR_MAX_ITER, lag=LAG, metric=eval_metric)
        savefig(f"graphs/data_modeling/mlp/{self.data_loader.file_tag}_mlp_{eval_metric}_study.png")
        show()

        # Best Model Performance Analysis
        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)
        figure()
        self.plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels, self.data_loader.file_tag)
        savefig(f'graphs/data_modeling/mlp/{self.data_loader.file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png')
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
        savefig(f"graphs/data_modeling/mlp/{self.data_loader.file_tag}_mlp_{eval_metric}_overfitting.png")

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
        savefig(f"graphs/data_modeling/mlp/{self.data_loader.file_tag}_mlp_{eval_metric}_loss_curve.png")