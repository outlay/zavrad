import pandas as pd
import glob
from algo.decision_tree import ClassificationTreeModel
from algo.random_forest import RandomForestModel
from algo.svm import SVMModel

def analyze_file(file):
    filename = file.split("\\")[1]
    print(f"Predicting: {filename}")
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    if not ('vwap_price' in df.columns and 'change_prev_close_percentage' in df.columns):
        df[['vwap_price', 'change_prev_close_percentage']] = df[['vwap_price', 'change_prev_close_percentage']].fillna(
            df[['vwap_price', 'change_prev_close_percentage']].median())
    df["tomorrow"] = df["last_price"].shift(-1)
    df["direction"] = (df["tomorrow"] > df["last_price"]).astype(int)
    df = df.iloc[:-1]

    horizons = [2, 5, 60, 250, 500]
    for horizon in horizons:
        rolling_avg = df['last_price'].rolling(window=horizon).mean()
        ratio = f"close_ratio_{horizon}"
        df[ratio] = df['last_price'] / rolling_avg

        trend = f"trend_{horizon}"
        df[trend] = df.shift(1)['direction'].rolling(window=horizon, min_periods=1).sum()
    df = df.dropna()

    models = [ClassificationTreeModel(df), RandomForestModel(df), SVMModel(df)]

    for model in models:
        model.fit()
        model.cross_validate_and_fit()
        model.predict()
        model.plot(filename.split(".")[0])


def main():
    files = glob.glob('./data/*.csv')
    for i, file in enumerate(files):
        analyze_file(file)
    print("")




if __name__ == "__main__":
    main()
