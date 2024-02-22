import yfinance as yf


msft = yf.Ticker("AMZN")

company_name = msft.info['longName']
print(company_name)