from flask import Flask, render_template, request, send_file
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)

def get_stock_data(ticker, period='1y', interval='1d'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def compute_ema(df, span):
    return df['Close'].ewm(span=span, adjust=False).mean()

def compute_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(df, short_span=12, long_span=26, rsi_window=14, rsi_buy_thresh=30, rsi_sell_thresh=70):
    df['EMA_short'] = compute_ema(df, short_span)
    df['EMA_long'] = compute_ema(df, long_span)
    df['RSI'] = compute_rsi(df, rsi_window)

    df['Signal'] = 0
    df.loc[df['EMA_short'] > df['EMA_long'], 'Signal'] = 1
    df.loc[df['EMA_short'] < df['EMA_long'], 'Signal'] = -1
    df['Position'] = df['Signal'].diff()

    buy_signals = (df['Position'] == 2) | ((df['Position'] == 1) & (df['RSI'] < rsi_buy_thresh))
    sell_signals = (df['Position'] == -2) | ((df['Position'] == -1) & (df['RSI'] > rsi_sell_thresh))

    df['Buy_Signal_Price'] = np.where(buy_signals, df['Close'], np.nan)
    df['Sell_Signal_Price'] = np.where(sell_signals, df['Close'], np.nan)

    return df

def plot_signals(df, ticker):
    plt.figure(figsize=(14,7))
    plt.plot(df['Close'], label='Close Price', alpha=0.5)
    plt.plot(df['EMA_short'], label='Short EMA', alpha=0.9)
    plt.plot(df['EMA_long'], label='Long EMA', alpha=0.9)
    plt.scatter(df.index, df['Buy_Signal_Price'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(df.index, df['Sell_Signal_Price'], marker='v', color='red', label='Sell Signal', s=100)
    plt.title(f"{ticker} Buy and Sell Signals (EMA + RSI Filter)")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        try:
            df = get_stock_data(ticker)
            df = generate_signals(df)
            plot_buf = plot_signals(df, ticker)
            return send_file(plot_buf, mimetype='image/png')
        except Exception as e:
            return f"<h3>Error: {e}</h3>"
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
