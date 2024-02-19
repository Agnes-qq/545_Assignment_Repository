import csv
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify, make_response
)
from werkzeug.exceptions import abort

from flaskr.db import get_db
from flaskr.get_data import get_stock_info, get_stock_news, get_comp_news
bp = Blueprint('views', __name__)


@bp.route('/')
def index():
    return render_template("index.html", page_data={"title": "MARKET INSIGHT: Home"})


@bp.route('/stock/')
@bp.route('/stock/<symbol>')
def show_stock_info(symbol = None):
    if symbol:       
        page_data = {"title": "MARKET INSIGHT: " + symbol, "inputValue": symbol}
        try:
            get_stock_info(symbol)
            with open('flaskr/chart/data/'+symbol.upper()+'.csv')  as csvfile:
                return render_template("detailed.html", page_data=page_data, symbol=symbol) 
        except:
            # see https://flask.palletsprojects.com/en/3.0.x/patterns/flashing/#flashing-with-categories
            flash('Stock symbol not found: '+symbol,'error') 
    else:
        page_data = {"title": "MARKET INSIGHT: Stock Information", "inputValue": ""}
    return render_template("detailed.html", page_data=page_data, )

'''def show_comp_summaries(symbol):
        from flaskr.get_data import get_stock_info, get_stock_news, get_comp_news
        get_stock_info(symbol)
        get_stock_news(symbol)
        get_comp_news(symbol)'''



@bp.route('/stock/pricing/<symbol>')
def retrieve_stock_prices(symbol):

    close_prices = []
    dates = []

    try:
        with open('flaskr/chart/data/'+symbol.upper()+'.csv')  as csvfile:
            stockreader = csv.DictReader(csvfile)
            for row in stockreader:
                dates.append(row["Date"])
                close_prices.append(row["Adj Close"])
            result = {
                "symbol": symbol,
                "dates": dates,
                "adjClosePrices": close_prices
            }
            return jsonify(result)
    except:
        return make_response(jsonify({'error': symbol+' - not found'}), 404)
