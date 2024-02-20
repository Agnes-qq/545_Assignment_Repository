# Flaskr
Flaskr application is leveraged in this project


## Run

Run the following commands to start the flaskr application.
$ flask --app flaskr run --debug


## Interaction with api_key
personal api_key is stored in my_flask/config.py, which is not commited to the repo.
And in the my_flask/get_data.py, we can 'import config', and then get access to the api_key stored in the config.py to pull out the stock data from Alpha Vantage.

##To access the stock information page for a particular stock
just input the stock symbol in the search bar and retrieve