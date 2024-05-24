from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV


class RandomForestModel:
    def __init__(self, df):
        self.X = df.drop(columns=['Date', 'direction', 'tomorrow'])
        self.Y = df['direction']
        self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(self.X,
                                                                                self.Y,
                                                                                test_size=0.33,
                                                                                random_state=1010)
        self.rf_model = None
        self.y_pred = None

    def fit(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1010)
        self.rf_model.fit(self.X_train, self.Y_train)

    def predict(self):
        self.y_pred = self.rf_model.predict(self.x_test)
        print("Precision score: " + str(precision_score(self.y_test, self.y_pred)))

    def plot(self, title):
        fig, ax = plt.subplots(figsize=(8, 8))
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.rf_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cijena pada", "Cijena raste"])
        disp.plot(ax=ax)
        ax.set_title(f"Nasumična šuma - {title}")
        precision = precision_score(self.y_test, self.y_pred)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)

        plt.figtext(0.5, 0.1,
                    f"Preciznost: {precision:.4f}\nTočnost: {accuracy:.4f}\nOdziv: {recall:.4f}",
                    ha="center", fontsize=12, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        plt.subplots_adjust(bottom=0.2)

        filename = f"{title}_nasumicna_suma.png"
        plt.savefig(f"./img/rf/{filename}")
        plt.close(fig)

    def cross_validate_and_fit(self):
        param_grid = [{
            'n_estimators': [5, 10, 15, 20, 50, 100],
            'max_depth': [2, 5, 7, 9, None],
            'min_samples_split': [10,50,100]
        }]

        optimal = GridSearchCV(
            RandomForestClassifier(),
            param_grid,
            cv=10,
            scoring='accuracy',
            verbose=0
        )
        optimal.fit(self.X_train, self.Y_train)
        print(optimal.best_params_)

        self.rf_model = RandomForestClassifier(n_estimators=optimal.best_params_['n_estimators'],
                                               max_depth=optimal.best_params_['max_depth'],
                                               min_samples_split=optimal.best_params_['min_samples_split'],
                                               random_state=1010)
        self.rf_model.fit(self.X_train, self.Y_train)
