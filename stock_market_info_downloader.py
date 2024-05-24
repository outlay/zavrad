import argparse
import csv

import yfinance as yf
import requests


# https://rest.zse.hr/web/Bvt9fe2peQ7pwpyYqODM/security-history/XZAG/HRKOEIRA0009/2024-05-01/2024-05-03/json

# ht ticker - HRHT00RA0005
# koei ticker - HRKOEIRA0009


# NVDA, ENR.F, HT,  KOEI,  VOO, VWCE
def download_data_yahoo(ticker, start, end, suffix):
    print(f'Downloading {ticker} data from {start} to {end}')
    data = yf.download(ticker, start=start, end=end)

    data = data.rename(columns={
        'Date': 'date',
        'Open': 'open_price',
        'High': 'high_price',
        'Low': 'low_price',
        'Close': 'last_price',
        'Volume': 'volume'
    })

    data['change_prev_close_percentage'] = data['last_price'].pct_change() * 100

    data['vwap_price'] = (data['volume'] * (data['high_price'] + data['low_price'] + data['last_price']) / 3).cumsum() / \
                         data['volume'].cumsum()

    data = data[['open_price', 'high_price', 'low_price', 'last_price', 'vwap_price', 'change_prev_close_percentage',
                 'volume']]

    data.to_csv(f'./data/{ticker + suffix}.csv')
    print(f'Data saved to ./data/{ticker}.csv')


def download_data_zse(ticker, start, end, suffix):
    print(f'Downloading {ticker} data from {start} to {end}')
    url = f'https://rest.zse.hr/web/Bvt9fe2peQ7pwpyYqODM/security-history/XZAG/{ticker}/{start}/{end}/json'
    response = requests.get(url)
    data = response.json()
    history = data['history']
    output_file = f'./data/{ticker+suffix}.csv'
    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ['date', 'open_price', 'high_price', 'low_price', 'last_price', 'vwap_price',
                      'change_prev_close_percentage', 'num_trades', 'volume', 'turnover']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for entry in history:
            writer.writerow({
                'date': entry['date'],
                'open_price': entry['open_price'],
                'high_price': entry['high_price'],
                'low_price': entry['low_price'],
                'last_price': entry['last_price'],
                'vwap_price': entry['vwap_price'],
                'change_prev_close_percentage': entry['change_prev_close_percentage'],
                'num_trades': entry['num_trades'],
                'volume': entry['volume'],
                'turnover': entry['turnover']
            })

    print(f'Data saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(description="Download yfinance,zse stock data")

    parser.add_argument('ticker', type=str, help='Stock ticker')
    parser.add_argument('--source', default='yfinance', type=str, help='Data source')
    parser.add_argument('--start', default=None, type=str, help='Start date')
    parser.add_argument('--end', default=None, type=str, help='End date')
    parser.add_argument('--suffix', default=None, type=str, help='File suffix')

    args = parser.parse_args()
    if args.source == 'yfinance':
        download_data_yahoo(args.ticker, args.start, args.end, args.suffix)
    else:
        download_data_zse(args.ticker, args.start, args.end, args.suffix)


if __name__ == '__main__':
    main()
