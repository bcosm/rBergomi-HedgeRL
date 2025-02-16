# Script for fetching SPY data for simulations.

import yfinance as yf
import pandas as pd
import argparse
import sys

# Main function for downloading SPY data.
def main():
    parser = argparse.ArgumentParser(description='Download historical SPY price data')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--days', type=int, default=1000, help='Number of days to download (default: 1000)')
    parser.add_argument('--output', type=str, default='historical_prices.csv', help='Output filename (default: historical_prices.csv)')
    args = parser.parse_args()
    
    ticker_symbol = args.ticker
    num_days = args.days
    output_filename = args.output
    
    buffer_factor = 1.5
    period_str = f"{int(num_days * buffer_factor)}d"
    
    print(f"Downloading {ticker_symbol} data...")
    try:
        data = yf.download(ticker_symbol, period=period_str)
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)
    
    if data.empty:
        print(f"Error: No data downloaded for {ticker_symbol}")
        sys.exit(1)
    
    closing_prices = data['Close']
    
    if len(closing_prices) > num_days:
        closing_prices = closing_prices[-num_days:]
    
    pd.DataFrame(closing_prices.values).to_csv(output_filename, header=False, index=False)
    
    print(f"Successfully saved {len(closing_prices)} days of {ticker_symbol} closing prices to {output_filename}")
    print(f"Date range: {closing_prices.index[0].date()} to {closing_prices.index[-1].date()}")

if __name__ == "__main__":
    main()
