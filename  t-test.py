import pandas as pd
from scipy.stats import ttest_ind

tickers = ["AAPL", "AMZN", "TSLA", "MSFT", "GOOGL", "NFLX", "JPM", "V", "GS"]
#"AAPL", "AMZN", "TSLA", "FB",
for t in tickers:
    # Load the CSV file
    file_path = f'./{t}_data.csv' # Replace with your file's path
    data = pd.read_csv(file_path)

    # Ensure the data contains no missing values for the required columns
    data = data[['headline_score', 'Adj Close Change']].dropna()

    # Divide data into two groups based on the headline score
    # You can define the threshold based on your analysis (e.g., positive vs. negative scores)
    threshold = 0
    group_positive = data[data['headline_score'] > threshold]['Adj Close Change']
    group_negative = data[data['headline_score'] <= threshold]['Adj Close Change']

    # Perform a t-test
    t_stat, p_value = ttest_ind(group_positive, group_negative)

    # Print the results
    print(t)
    print("T-Statistic:", t_stat)
    print("P-Value:", p_value)

    # Interpret results
    if p_value < 0.05:
        print("There is a statistically significant relationship between headline_score and Adj Close Change.")
    else:
        print("No statistically significant relationship was found between headline_score and Adj Close Change.")