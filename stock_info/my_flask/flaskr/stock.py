from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.db import get_db

bp = Blueprint('stock', __name__)


'''@bp.route("/")
def index():
    db = get_db()
    posts = db.execute(
        "SELECT symbol, title, body, created, author_id, username"
        " FROM post p JOIN user u ON p.author_id = u.id"
        " ORDER BY created DESC"
    ).fetchall()
    return render_template("stock/index.html", posts=posts)'''


@bp.route("/", methods=("GET", "POST"))
def search_page():
    if request.method == "POST":
        symbol = request.form["title"]
        body = request.form["body"]
        error = None

        if not symbol:
            error = "Symbol is required."

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                "INSERT INTO stock (symbol, body) VALUES (?, ?)",
                (symbol, body),
            )
            db.commit()
            return redirect(url_for("stock.index"))

    return render_template("stock/detailed.html")


@bp.route("/detailed", methods=("GET","POST"))
def show_stock_info(symbol):

    '''    
    try (open the csv data): close_price, dates, symbol
    else: flask(error)
    '''

    if request.method == "POST":
        title = symbol
        body = symbol["body"]
        error = None

        if not title:
            error = "Title is required."

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                "UPDATE post SET title = ?, body = ?", (symbol, body )
            )
            db.commit()
            return redirect(url_for("blog.index"))

    return render_template("/index.html")

'''
def show_stock_info(symbol):
    return a dict contain all the info: close_price, dates, symbol
'''