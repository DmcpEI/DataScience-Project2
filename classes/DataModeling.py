from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show, title
from typing import Literal
from sklearn.neighbors import KNeighborsClassifier

from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart, plot_evaluation_results, plot_multiline_chart


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

        return best_model, best_params

    def naive_bayes(self):

        trnX: ndarray = self.X_train.values
        tstX: ndarray = self.X_test.values
        trnY: array = self.y_train.values
        tstY: array = self.y_test.values
        labels: list = list(self.y_train.unique())
        labels.sort()
        vars = self.X_train.columns.to_list()

        print(f"Train#={len(trnX)} Test#={len(tstX)}")
        print(f"Labels={labels}")

        # Accuracy
        figure(figsize=(7, 5))
        best_model, params = self.naive_Bayes_study(trnX, trnY, tstX, tstY)
        savefig(f"graphs/data_modeling/nb/{self.data_loader.file_tag}_nb_accuracy_study.png")
        show()

        # Recall
        figure(figsize=(7, 5))
        best_model, params = self.naive_Bayes_study(trnX, trnY, tstX, tstY, "recall")
        savefig(f"graphs/data_modeling/nb/{self.data_loader.file_tag}_nb_recall_study.png")
        show()

        # Performance Analysis
        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)
        figure()
        plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
        savefig(f'graphs/data_modeling/nb/{self.data_loader.file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
        show()

    def knn_study(
            self, trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max: int = 19, lag: int = 2, metric='accuracy'
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
        print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')
        plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric,
                             percentage=True)

        return best_model, best_params

    def knn(self):

        eval_metric = 'accuracy'

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
        best_model, params = self.knn_study(trnX, trnY, tstX, tstY, k_max=25, metric=eval_metric)
        savefig(f'graphs/data_modeling/knn/{self.data_loader.file_tag}_knn_{eval_metric}_study.png')
        show()

        # Best Model Performance Analysis
        prd_trn: array = best_model.predict(trnX)
        prd_tst: array = best_model.predict(tstX)
        figure()
        plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
        savefig(f'graphs/data_modeling/knn/{self.data_loader.file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png')
        show()

        # Overfitting Study
        distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
        K_MAX = 25
        kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
        y_tst_values: list = []
        y_trn_values: list = []
        acc_metric: str = "accuracy"
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=distance)
            clf.fit(trnX, trnY)
            prd_tst_Y: array = clf.predict(tstX)
            prd_trn_Y: array = clf.predict(trnX)
            y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
            y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

        figure()
        plot_multiline_chart(
            kvalues,
            {"Train": y_trn_values, "Test": y_tst_values},
            title=f"KNN overfitting study for {distance}",
            xlabel="K",
            ylabel=str(eval_metric),
            percentage=True,
        )
        savefig(f"graphs/data_modeling/knn/{self.data_loader.file_tag}_knn_overfitting.png")
        show()

