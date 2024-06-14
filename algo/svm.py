from sklearn.metrics import ConfusionMatrixDisplay, precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class SVMModel:
    def __init__(self, df):
        self.X = df.drop(columns=['Date', 'direction', 'tomorrow'])
        self.Y = df['direction']
        self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(self.X,
                                                                                self.Y,
                                                                                test_size=0.33,
                                                                                random_state=1010)
        self.X_scaled = None
        self.svr_model = None
        self.y_pred = None

    def fit(self):
        self.X_scaled = scale(self.X_train)
        svr_model = SVC(kernel='rbf', C=100, gamma=0.1)
        svr_model.fit(self.X_scaled, self.Y_train)
        self.svr_model = svr_model

    def predict(self):
        x_test_scaled = scale(self.x_test)

        self.y_pred = self.svr_model.predict(x_test_scaled)
        print("Precision score: " + str(precision_score(self.y_test, self.y_pred)))

    def plot(self, title):
        fig, ax = plt.subplots(figsize=(8, 8))
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.svr_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cijena pada", "Cijena raste"])
        disp.plot(ax=ax)
        ax.set_title(f"SVM - {title}", fontsize=20)
        ax.set_xlabel('Predviđeno', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10)
        precision = precision_score(self.y_test, self.y_pred)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        for text in disp.text_.ravel():
            text.set_fontsize(24)


        plt.figtext(0.80, 0.03,
                    f"Preciznost: {precision:.4f}\nTočnost: {accuracy:.4f}\nOdziv: {recall:.4f}",
                    ha="center", fontsize=24, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        plt.subplots_adjust(bottom=0.2)

        filename = f"{title}_svm.png"
        plt.savefig(f"./img/svm/{filename}")
        plt.close(fig)

    def cross_validate_and_fit(self):
        param_grid = [{
            'C': [0.1, 0.5, 1, 10, 100],
            'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }]

        optimal = GridSearchCV(
            SVC(),
            param_grid,
            cv=5,
            scoring='precision',
            verbose=0
        )
        optimal.fit(self.X_scaled, self.Y_train)
        print(optimal.best_params_)

        self.svr_model = SVC(kernel='rbf', C=optimal.best_params_['C'], gamma=optimal.best_params_['gamma'])
        self.svr_model.fit(self.X_scaled, self.Y_train)
