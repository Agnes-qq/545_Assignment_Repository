import requests
import pandas as pd
import config as config
import os
path = os.getcwd()

def get_stock_info(symbol):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol='+symbol+'&apikey='+config.api_key
    r = requests.get(url)
    data = r.json()

    values = []
    headers = []
    for i in data['Time Series (Daily)']:
        headers.append(i)
        #col_name = i.split(". ")[1]
        values.append(data['Time Series (Daily)'][i])
    df = pd.DataFrame(values)
    df = df.set_axis(headers, axis=0)
    df.reset_index(inplace= True)
    

    cols = ['Date','Open','High','Low','Close','Adj Close','Volume','Dividend amount', 'Split coefficient']
    df.columns = cols
    df.set_index('Date', inplace=True)
    df.drop(['Dividend amount', 'Split coefficient'], axis = 1,inplace=True)
    df[::-1].to_csv(path+"/flaskr/chart/data/"+symbol+".csv")
    return df
get_stock_info('IOVA')


def get_stock_news(symbol):
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers='+symbol+'&apikey='+config.api_key
        #  https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM     &apikey=demo            &datatype=csv
    r = requests.get(url)
    data = r.json()
    summaries = []
    titles = []
    url = []
    for i in range(5):
        titles.append(data['feed'][i]['title'])
        summaries.append(data['feed'][i]['summary'])
        url.append(data['feed'][i]['url'])
    df_news = pd.DataFrame([titles,summaries,url]).T
    df_news.to_csv(path+"/flaskr/chart/data/"+symbol+"_summaries.csv")
    return df_news



def get_comp_info(symbol):
    url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol='+symbol+'&apikey='+config.api_key
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
    comp_info.to_csv(path+"/flaskr/chart/data/"+symbol+"_info.csv")
    return comp_info

