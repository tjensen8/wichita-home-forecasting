import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from utils.environment_vars import load_yaml, get_bls_key
from MarketData import FredQuery
import os



# will be forcasting housing prices in wichita to anticipate housing moves
# over the next 12 months

if __name__ == "__main__":
    series_ids = load_yaml(os.path.abspath("./bls_series.yaml"))
    fred = FredQuery(get_bls_key())

    fred.get_market_data_df(series_ids['BLS_SERIES'], resample="Q")

    fred.plot_market_data()

    market_data = fred.market_data_df()

    # display target variable
    market_data["House Price Index, Wichita"].plot()
    plt.title("Index of Historical Home Prices, Wichita KS")
    plt.ylabel("Index Value")
    plt.xlabel("Timeline (Months)")
    plt.savefig('./figures/house_price_history_ict.png')
    plt.close()

    forecast_months = [1, 2, 3, 4, 5, 6]
    results = [market_data["House Price Index, Wichita"][-1]]

    for months in forecast_months:
        ### shift data to predict next time instance
        market_data["House Price Index, Wichita"] = market_data[
            "House Price Index, Wichita"
        ].shift(months)
        market_data.dropna(inplace=True)

        print(market_data.corr()["House Price Index, Wichita"])

        scaler = StandardScaler()

        Y = market_data["House Price Index, Wichita"]
        X = market_data.drop("House Price Index, Wichita", axis=1)

        lr = LinearRegression()

        # breaking into train and test splits
        x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True)

        # training
        lr.fit(x_train, y_train)

        y_pred_train = lr.predict(x_train)

        train_score_r2 = lr.score(x_train, y_train)
        train_score_mae = mean_absolute_error(y_train, y_pred_train)
        print("Training Scores:")
        print("R2: ", np.round(train_score_r2, 3))
        print("MAE: ", np.round(train_score_mae, 3))

        # testing
        y_pred_test = lr.predict(x_test)

        test_score_r2 = lr.score(x_test, y_test)
        train_score_mae = mean_absolute_error(y_test, y_pred_test)
        print("Test Scores")
        print("R2: ", np.round(test_score_r2, 3))
        print("MAE: ", np.round(train_score_mae,3))

        residuals = y_test - y_pred_test

        plt.scatter(residuals, y_pred_test)
        plt.title(f"Residuals - {months} Forecast")
        plt.xlabel("Residuals")
        plt.ylabel("Prediction")
        plt.savefig(f'./figures/residuals-{months}-month-forecast.png')
        plt.close()

        print("\nPrediction: ", lr.predict(X.iloc[-1].values.reshape([1, -1])))
        results.append(lr.predict(X.iloc[-1].values.reshape([1, -1])))

    results = pd.DataFrame(results)
    plt.title("Forecasts")
    #plt.plot(results.pct_change() * 100)
    plt.plot(results)
    plt.ylabel("Forecasted Index Values")
    plt.xlabel("Months Forecasted Out")
    plt.savefig('./figures/forecast_results.png')
    plt.close()
