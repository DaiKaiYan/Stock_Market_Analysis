import analysis
import matplotlib.pyplot as plt

# Create stock instances
apple = analysis.Stock('AAPL', 'Apple')
microsoft = analysis.Stock('MSFT', 'Microsoft')
google = analysis.Stock('GOOG', 'Google')
amazon = analysis.Stock('AMZN', 'Amazon')

# Download stock data
for stock in [apple, microsoft, google, amazon]:
    stock.download_data()

# Print summary statistics for Apple stock data
print(apple.df.describe())

# Visualize adjusted closing prices for multiple stocks
visualization = analysis.Visualize4Stocks([apple, microsoft, google, amazon])
visualization.plot_line('Adj Close')

# Calculate and plot moving averages
moving_average_days = [10, 20, 50]
for ma in moving_average_days:
    for stock in [apple, microsoft, google, amazon]:
        stock.compute_moving_average(ma)

visualization.plot_lines(['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days'])

# Calculate daily returns
for stock in [apple, microsoft, google, amazon]:
    stock.compute_return()

# Plot histogram of daily returns
visualization.plot_histogram('Daily Return')

# Create attribute data for daily returns
return_data = analysis.AttributeData('Daily Return')

# Add return data for each stock
for stock in [apple, microsoft, google, amazon]:
    return_data.add_stock(stock)

# Plot correlation between two stocks
return_data.plot_correlation_one_pair(apple, microsoft)

# Plot correlation for all stocks
return_data.plot_correlation()

# Plot correlation heatmap
return_data.plot_correlation_heatmap()

# Plot mean and standard deviation
return_data.plot_mean_std()

# Create Nvidia stock instance and download data
nvidia_stock = analysis.Stock('NVDA', 'Nvidia')
nvidia_stock.download_data(start='2012-01-01')
print(nvidia_stock.df.describe())

# Create prediction model
prediction_model = analysis.PredictionModel(nvidia_stock, 'Close')

# Plot Nvidia stock closing prices
nvidia_stock.plot_line('Close')

# Split data into training and testing sets
x_train, y_train, x_test, y_test = prediction_model.train_test_split(train_pct=0.8)

# Train the model
prediction_model.train(x_train, y_train)

# Make predictions
predictions = prediction_model.predict(x_test)

# Calculate RMSE
prediction_model.calculate_rmse(predictions, y_test)

# Plot predictions
prediction_model.plot_predictions(predictions)

# Show all plots
plt.show()