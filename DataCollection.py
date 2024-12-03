import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
# from langchain_openai import ChatOpenAI

from transformers import pipeline
import finnhub
import time

def get_stock(stock,start_date=datetime.today().date() - timedelta(days=365),end_date = datetime.today().date() - timedelta(days=1)):    
    stock_data = yf.download(stock,start=start_date,end=end_date)
    stock_data = stock_data.drop(columns={'Close'})

    stock_data['Prev Day'] = stock_data['Adj Close'].shift(1)
    #Do the closing price of today will be hiher than the closing price from the previous day
    stock_data['Target'] = (stock_data['Adj Close'] > stock_data['Prev Day']).astype(int)
    stock_data['Adj Close Change'] = stock_data['Adj Close'].pct_change() * 100

    stock_data = stock_data.reset_index()

    return stock_data

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def safe_sentiment(text):
    try:
        return sentiment_pipeline(text)[0]
    except Exception as e:
        print(f"Error processing text: {e}")
        # Optionally, return neutral/default sentiment
        return {'label': 'NEUTRAL', 'score': 0}
    
def get_news(stock,start_date=datetime.today().date() - timedelta(days=365),end_date=datetime.today().date() - timedelta(days=1)):    
    # news
    finnhub_client = finnhub.Client(api_key="cnl0n1pr01qjvabn3kngcnl0n1pr01qjvabn3ko0")
    #news = pd.DataFrame(finnhub_client.company_news(stock, _from=start_date, to=end_date)).drop(columns=['id','image','category'])

    # DataFrame to store all news
    all_news = pd.DataFrame()

    # Loop over the date range in steps of 5 days
    current_date = start_date
    while current_date < end_date:
        _to = current_date + timedelta(days=5)
        if _to > end_date:
            _to = end_date

        # Fetch the news
        news_data = pd.DataFrame(finnhub_client.company_news(stock, _from=current_date.strftime('%Y-%m-%d'), to=_to.strftime('%Y-%m-%d')))
        if not news_data.empty:
            news_data = news_data.drop(columns=['id', 'image', 'category'])
            news_data['datetime'] = pd.to_datetime(news_data['datetime'], unit='s').dt.strftime('%Y-%m-%d')
            all_news = pd.concat([news_data.iloc[::-1], all_news], ignore_index=True)
        
        # Update the current date
        current_date = _to

        #sleep to avoid reaching limit
        time.sleep(1)

    # Save the DataFrame to a CSV file
    #all_news.to_csv(f'./{stock}_news.csv', index=False)

    #Sentiment
    #sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    #news_headline = all_news.groupby(['datetime'])['headline'].agg(lambda x: " ".join(x)).reset_index().sort_values('datetime',ascending=True)

     # Assuming 'summary' column in 'pll_news_semtiment' DataFrame contains the input sequences
    #news_headline['truncated_headline'] = news_headline['headline'].apply(lambda x: truncate_sequence(x))
    
    # # Now you can apply the sentiment analysis pipeline to the truncated sequences
    # pll_news_semtiment['summary_sentiment'] = pll_news_semtiment['truncated_summary'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    # pll_news_semtiment['summary_sentiment_score'] = pll_news_semtiment['truncated_summary'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    
    all_news['headline_sentiment'] = all_news['headline'].apply(lambda x: safe_sentiment(x) )
    #all_news['headline_sentiment_score'] = all_news['headline'].apply(lambda x: safe_sentiment(x)['score'] )
    #all_news['summary_sentiment'] = all_news['summary'].apply(lambda x: safe_sentiment(x) )
    #all_news['summary_sentiment_score'] = all_news['summary'].apply(lambda x: safe_sentiment(x)['score'] )

    all_news['adjusted_headline_score'] = all_news.apply(
        lambda x: -x['headline_sentiment']['score'] if x['headline_sentiment']['label'] == 'NEGATIVE' else x['headline_sentiment']['score'],
        axis=1
    )
    # all_news['adjusted_summary_score'] = all_news.apply(
    #     lambda x: -x['summary_sentiment']['score'] if x['summary_sentiment']['label'] == 'negative' else x['summary_sentiment']['score'],
    #     axis=1
    # )

    # Group by 'datetime' and sum the adjusted scores
    grouped_scores = all_news.groupby('datetime').agg({
        'adjusted_headline_score': 'mean'
        #'adjusted_summary_score': 'sum'
    }).reset_index()

    # Rename columns for clarity
    grouped_scores.columns = ['datetime', 'headline_score']

    return grouped_scores

def stock_and_news(stock,news):
    stock['Date'] = pd.to_datetime(stock['Date'])
    news['datetime'] = pd.to_datetime(news['datetime'])
    master_df = stock.merge(right=news,left_on='Date',right_on='datetime',how='left').drop(columns=['datetime'])
    return master_df


tickers = ["AAPL", "AMZN", "TSLA", "MSFT", "GOOGL", "NFLX", "JPM", "V", "GS"]
#"AAPL", "AMZN", "TSLA", "FB",
for t in tickers:
    print("start collecting ", t)
    print("getting stock data")
    stock_data = get_stock(t)
    print("getting news data")
    stock_news = get_news(t)
    stock_news = stock_and_news(stock_data,stock_news)
    stock_news['headline_score'] = stock_news['headline_score'].fillna(0)
    stock_news.to_csv(f'./{t}_data.csv', index=False)