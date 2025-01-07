import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2


class UAVWiFiClassifier:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self.load_data()
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None

    def load_data(self):
        data = pd.read_excel(self.filepath)
        print(data.info())
        return data

    def preprocess_data(self):
        self.X = self.data.drop('target', axis=1)
        self.Y = self.data['target']
        print('X shape = ', self.X.shape)
        print('Y shape = ', self.Y.shape)

    def split_data(self, test_size=0.30, random_state=1):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state)
        print('X train shape = ', self.X_train.shape)
        print('X test shape = ', self.X_test.shape)
        print('Y train shape = ', self.Y_train.shape)
        print('Y test shape = ', self.Y_test.shape)

    def train_model(self):
        self.model = SVC(gamma='scale', random_state=0)
        self.model.fit(self.X_train, self.Y_train)
        score = self.model.score(self.X_test, self.Y_test)
        print('Support Vector Classification Model Score = ', score)

    def plot_boxplots(self):
        first_10_columns = self.X.iloc[:, 0:5]
        plt.figure(figsize=(10, 5))
        first_10_columns.boxplot()
        plt.title("Boxplot of Original Features")
        plt.show()

    def scale_features(self):
        self.X_scaled = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        first_10_columns = self.X_scaled.iloc[:, 0:5]
        plt.figure(figsize=(10, 5))
        first_10_columns.boxplot()
        plt.title("Boxplot of Scaled Features")
        plt.show()

    def select_best_features(self, k=10):
        best_input_columns = SelectKBest(chi2, k=k).fit(self.X_scaled, self.Y)
        sel_index = best_input_columns.get_support()
        best_X = self.X_scaled.loc[:, sel_index]
        feature_selected = best_X.columns.values.tolist()
        print("The best 10 features selected are:", feature_selected)
        return best_X

    def re_split_and_train(self):
        best_X = self.select_best_features()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(best_X, self.Y, test_size=0.30, random_state=1)
        print('X train shape = ', self.X_train.shape)
        print('X test shape = ', self.X_test.shape)
        print('Y train shape = ', self.Y_train.shape)
        print('Y test shape = ', self.Y_test.shape)

        self.model = SVC(gamma='auto', random_state=0)
        self.model.fit(self.X_train, self.Y_train)
        score = self.model.score(self.X_test, self.Y_test)
        print('Support Vector Classification Model Score (after feature selection) = ', score)


if __name__ == "__main__":
    classifier = UAVWiFiClassifier(filepath='UAV_WiFi.xlsx')
    classifier.preprocess_data()
    classifier.split_data()
    classifier.train_model()
    classifier.plot_boxplots()
    classifier.scale_features()
    classifier.re_split_and_train()
