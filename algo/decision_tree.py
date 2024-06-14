from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


class ClassificationTreeModel:
    def __init__(self, df):
        self.X = df.drop(columns=['Date', 'direction', 'tomorrow'])
        self.Y = df['direction']
        self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(self.X,
                                                                                self.Y,
                                                                                test_size=0.33,
                                                                                random_state=1010)
        self.cltree_model = None
        self.y_pred = None

    def fit(self):
        self.cltree_model = DecisionTreeClassifier(random_state=1010)
        self.cltree_model.fit(self.X_train, self.Y_train)

    def predict(self):
        self.y_pred = self.cltree_model.predict(self.x_test)
        print("Precision score: " + str(precision_score(self.y_test, self.y_pred)))

    def plot(self, title):
        fig, ax = plt.subplots(figsize=(8, 8))
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.cltree_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cijena pada", "Cijena raste"])
        disp.plot(ax=ax)
        ax.set_title(f"Stablo odluke - {title}", fontsize=20)
        ax.set_xlabel('Predviđeno', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10)
        precision = precision_score(self.y_test, self.y_pred)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        for text in disp.text_.ravel():
            text.set_fontsize(32)

        plt.figtext(0.80, 0.03,
                    f"Preciznost: {precision:.4f}\nTočnost: {accuracy:.4f}\nOdziv: {recall:.4f}",
                    ha="center", fontsize=24, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        plt.subplots_adjust(bottom=0.2)

        filename = f"{title}_stablo_odluke.png"
        plt.savefig(f"./img/ct/{filename}")
        plt.close(fig)

    def cross_validate_and_fit(self):
        param_grid = {'criterion': ['gini', 'entropy'],
                     'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
        optimal = GridSearchCV(
            DecisionTreeClassifier(),
            param_grid,
            cv=10,
            scoring='accuracy',
            verbose=0
        )
        optimal.fit(self.X_train, self.Y_train)
        print(optimal.best_params_)

        self.cltree_model = DecisionTreeClassifier(criterion=optimal.best_params_['criterion'], max_depth=optimal.best_params_['max_depth'],
                                                   random_state=1010)
        self.cltree_model.fit(self.X_train, self.Y_train)
