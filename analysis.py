# Module for stock market analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


class Stock:
    '''Gather the data for a single stock in a dataframe.'''

    def __init__(self, ticker, company_name):
        self.ticker = ticker
        self.company_name = company_name
        self.df = np.nan

    def download_data(self, num_years=1, start=np.nan):
        end = datetime.now()
        if start != start:
            start = datetime(end.year - num_years, end.month, end.day)
        else:
            start = pd.to_datetime(start)
        self.df = yf.download(self.ticker, start, end)
        self.df['company_name'] = self.company_name

    def compute_return(self):
        self.df['Daily Return'] = self.df['Adj Close'].pct_change()

    def compute_moving_average(self, ma_day):
        column_name = f"MA for {ma_day} days"
        self.df[column_name] = self.df['Adj Close'].rolling(ma_day).mean()

    def plot_line(self, column_name):
        '''Plot the trend of an attribute of the stock.'''
        plt.figure(figsize=(16, 6))
        plt.title(column_name)
        plt.plot(self.df[column_name])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel(column_name, fontsize=18)
        plt.show()


class AttributeData:
    '''Gather and visualize the values of a single attribute of multiple stocks in a dataframe.'''

    def __init__(self, column_name):
        self.column_name = column_name
        self.stock_list = []
        self.df = np.nan

    def add_stock(self, stock):
        '''Add the data of a stock to the attribute dataframe.'''
        self.stock_list.append(stock)
        if len(self.stock_list) == 1:
            self.df = stock.df[[self.column_name]].rename(columns={self.column_name: stock.company_name})
        else:
            self.df = self.df.join(stock.df[[self.column_name]].rename(columns={self.column_name: stock.company_name}),
                                   how='inner')

    def plot_correlation_one_pair(self, stock1, stock2):
        '''Compare the values of an attributes of two stocks using jointplot.'''
        df = self.df[[stock1.company_name, stock2.company_name]]
        sns.jointplot(x=stock1.company_name, y=stock2.company_name, data=df, kind='scatter')

    def plot_correlation(self):
        '''Compare the values of an attributes of each pair of stocks using various plots.'''
        return_fig = sns.PairGrid(self.df.dropna())
        # Scatter plots in the upper triangle
        return_fig.map_upper(plt.scatter, color='purple')
        # Kde plots in the lower triangle
        return_fig.map_lower(sns.kdeplot, cmap='cool_d')
        # Histograms in the diagonal
        return_fig.map_diag(plt.hist, bins=30)

    def plot_correlation_heatmap(self):
        '''Get numerical values for the correlation between stocks in a heatmap.'''
        sns.heatmap(self.df.corr(), annot=True, cmap='summer')
        plt.title('Correlation of %s' % self.column_name)

    def plot_mean_std(self):
        '''Scatter the means and standard deviations of an attribute of each stock.'''
        self.df = self.df.dropna()
        area = np.pi * 20
        plt.figure(figsize=(10, 8))
        plt.scatter(self.df.mean(), self.df.std(), s=area)
        plt.xlabel('Mean')
        plt.ylabel('Standard Deviation')

        for label, x, y in zip(self.df.columns, self.df.mean(), self.df.std()):
            plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom',
                         arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))


class Visualize4Stocks:
    '''Visualization for a list of 4 stocks.'''

    def __init__(self, stock_list):
        assert len(stock_list) == 4, "The number of stocks isn't 4. Please input a list of 4 stocks."
        self.stock_list = stock_list

    def plot_line(self, column_name):
        '''Plot 4 line graphs, one for each stock.'''
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=1.25, bottom=1.2)

        for i, company in enumerate(self.stock_list, 1):
            plt.subplot(2, 2, i)
            company.df[column_name].plot()
            plt.ylabel(column_name)
            plt.xlabel(None)
            plt.title(f"{column_name} of {company.company_name}")

        plt.tight_layout()

    def plot_lines(self, column_name_l):
        '''Plot 4 line graphs with multiple lines in each.'''

        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        for i, company in enumerate(self.stock_list, 1):
            company.df[column_name_l].plot(ax=axes[(i - 1) // 2, i - 1 - (i - 1) // 2 * 2])
            axes[(i - 1) // 2, i - 1 - (i - 1) // 2 * 2].set_title(company.company_name)

        fig.tight_layout()

    def plot_histogram(self, column_name):
        '''Plot 4 histograms, one for each stock.'''

        plt.figure(figsize=(12, 9))

        for i, company in enumerate(self.stock_list, 1):
            plt.subplot(2, 2, i)
            company.df[column_name].hist(bins=50)
            plt.xlabel(column_name)
            plt.ylabel('Counts')
            plt.title(company.company_name)

        plt.tight_layout()


class PredictionModel:
    '''Predict next-day value using past time-series data.'''
    def __init__(self, stock, column_name):
        self.stock = stock
        self.column_name = column_name
        self.data = stock.df[column_name].values
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_scaled = self.scaler.fit_transform(self.data.reshape(-1, 1))
        self.model = None

    def train_test_split(self, train_pct=0.95, train_days=60):
        '''Split data into training and testing datasets.'''
        train_size = int(len(self.data_scaled) * train_pct)
        train_data = self.data_scaled[:train_size]
        test_data = self.data_scaled[train_size:]

        x_train, y_train = [], []
        for i in range(train_days, len(train_data)):
            x_train.append(train_data[i - train_days:i, 0])
            y_train.append(train_data[i, 0])

        x_test, y_test = [], []
        for i in range(train_days, len(test_data)):
            x_test.append(test_data[i - train_days:i, 0])
            y_test.append(test_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_train, y_train, x_test, y_test

    def train(self, x_train, y_train):
        '''Train the model.'''
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(x_train, y_train, epochs=50, batch_size=32)

    def predict(self, x_test):
        '''Get the model-predicted values.'''
        predictions = self.model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions

    def calculate_rmse(self, y_pred, y_true):
        '''Get the root mean squared error (RMSE).'''
        return np.sqrt(np.mean((y_pred - y_true) ** 2))

    def plot_predictions(self, predictions):
        '''Compare predicted and true values in a line graph.'''
        plt.figure(figsize=(16, 6))
        # Plot actual data
        plt.plot(self.stock.df[self.column_name].values, color='#87CEFA', label='Actual')
        # Plot predicted data
        plt.plot(range(len(self.stock.df[self.column_name].values) - len(predictions),
                       len(self.stock.df[self.column_name].values)),
                 predictions, color='#FFB6C1', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.tight_layout()
