import requests
import pandas as pd
import json
import csv
#UDJ8FEEUYB4NYVJ6
def get_stock_info(symbol):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol='+symbol+'&apikey=PTZRDMMS8UYGPQ7G'
    r = requests.get(url)
    data = r.json()
    print(data)

    values = []
    headers = []
    for i in data['Time Series (Daily)']:
        headers.append(i)
        #col_name = i.split(". ")[1]
        values.append(data['Time Series (Daily)'][i])
    df = pd.DataFrame(values)
    df = df.set_axis(headers, axis=0)
    df.reset_index(inplace= True)

    cols = ['Date']
    for i in df.columns[1:]:
        cols_name = i.split(". ")[1]
        cols.append(cols_name.capitalize())
    df.columns = cols
    df.drop(['Dividend amount', 'Split coefficient'], axis = 1,inplace=True)

    return df
data = get_stock_info('AAPL')

symbol = "AAPL"
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol='+symbol+'&apikey=PTZRDMMS8UYGPQ7G'
r = requests.get(url)
data = r.json()
print(data)

values = []
headers = []
for i in data['Time Series (Daily)']:
    headers.append(i)
    #col_name = i.split(". ")[1]
    values.append(data['Time Series (Daily)'][i])
df = pd.DataFrame(values)
df = df.set_axis(headers, axis=0)
df.reset_index(inplace= True)

cols = ['Date']
for i in df.columns[1:]:
    cols_name = i.split(". ")[1]
    cols.append(cols_name.capitalize())
df.columns = cols
df.drop(['Dividend amount', 'Split coefficient'], axis = 1,inplace=True)


def get_stock_news(symbol):
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers='+symbol+'&apikey=PTZRDMMS8UYGPQ7G'
        #  https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM     &apikey=demo            &datatype=csv
    r = requests.get(url)
    data = r.json()
    summaries = []
    titles = []
    for i in range(5):
        titles.append(data['feed'][i]['title'])
        summaries.append(data['feed'][i]['summary'])
    df_news = pd.DataFrame([titles,summaries]).T
    df_news.to_csv(f'/Users/ruoqiyan/Documents/fintech-512-assignments/week5/stock_info/my_flask/flaskr/chart/data/{symbol}_summaries.csv')
    return df_news
get_stock_news('AAPL')




def get_comp_news(symbol):
    url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol='+symbol+'&apikey=PTZRDMMS8UYGPQ7G'
    r = requests.get(url)
    data = r.json()
    info_name= ['Name',
    'Symbol',
    'Exchange',
    'Sector', 
    'Industry',
    'MarketCapitalization' ,
    'PERatio' ,
    'EPS',
    'DividendPerShare' ,
    'DividendYield',
    '52WeekHigh',
    '52WeekLow']
    info_dict = {}
    for i in info_name:
        info_dict[i] = data[i]
    comp_info = pd.DataFrame(info_dict,index = [0])
    comp_info.to_csv(f'/Users/ruoqiyan/Documents/fintech-512-assignments/week5/stock_info/my_flask/flaskr/chart/data/{symbol}_summaries.csv')
    return comp_info







'''
data_price = data['Time Series (5min)']
current_p = 
previous_p = 
close_p = 
open_p =
volume = 

news_ = {1: {'headline': , 'summary': }}   #5
'''