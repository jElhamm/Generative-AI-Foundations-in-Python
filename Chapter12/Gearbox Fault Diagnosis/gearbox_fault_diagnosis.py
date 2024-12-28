import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay


class FaultPredictionModel:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.models = {}
    
    def load_data(self):
        data = pd.read_excel(self.filepath)
        print(data.head(10))
        print(data.info())
        return data

    def explore_data(self):
        data_stat = self.data.describe()
        print(data_stat)
        data_stat_cat = self.data.astype('object').describe()
        print(data_stat_cat)

        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        sns.boxplot(ax=axes[0], x='state', y='a1', data=self.data)
        sns.boxplot(ax=axes[1], x='state', y='a2', data=self.data)
        plt.ylim(-40, 40)
        plt.show()

    def preprocess_data(self):
        self.X = self.data.drop('state', axis=1)
        self.Y = self.data['state']
        print('X shape = ', self.X.shape)
        print('Y shape = ', self.Y.shape)

    def split_data(self, test_size=0.30, random_state=1):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state)
        print('X train shape = ', self.X_train.shape)
        print('X test shape = ', self.X_test.shape)
        print('Y train shape = ', self.Y_train.shape)
        print('Y test shape = ', self.Y_test.shape)

    def train_model(self, model_name, model):
        model.fit(self.X_train, self.Y_train)
        score = model.score(self.X_test, self.Y_test)
        self.models[model_name] = model
        print(f'{model_name} Model Score = ', score)
        return score

    def plot_decision_boundary(self, model_name):
        model = self.models[model_name]
        ax = DecisionBoundaryDisplay.from_estimator(
            model, self.X_train, response_method="predict", alpha=0.5)
        ax.ax_.scatter(self.X_train.iloc[:, 0], self.X_train.iloc[:, 1], c=self.Y_train, edgecolor="k")
        plt.title(f'Decision Boundary for {model_name}')
        plt.show()


if __name__ == "__main__":
    fault_model = FaultPredictionModel(filepath='fault.dataset.xlsx')

    fault_model.explore_data()
    fault_model.preprocess_data()
    fault_model.split_data()

    lr_model = LogisticRegression(random_state=0)
    fault_model.train_model("Logistic Regression", lr_model)
    fault_model.plot_decision_boundary("Logistic Regression")

    rf_model = RandomForestClassifier(max_depth=2, random_state=0)
    fault_model.train_model("Random Forest", rf_model)
    fault_model.plot_decision_boundary("Random Forest")

    mlp_model = MLPClassifier(random_state=1, max_iter=300)
    fault_model.train_model("Artificial Neural Network", mlp_model)
    fault_model.plot_decision_boundary("Artificial Neural Network")

    kn_model = KNeighborsClassifier(n_neighbors=2)
    fault_model.train_model("K-nearest neighbors", kn_model)
    fault_model.plot_decision_boundary("K-nearest neighbors")
