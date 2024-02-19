# Flaskr
The basic blog app built in the Flask [tutorial](https://flask.palletsprojects.com/tutorial/).


## Run
Ensure you have activated your class virtual environment::

    $ . ../venv/bin/activate

Run the following commands to start the flaskr application.  Note: init-db only needs to be performed once.

```bash
$ flask --app flaskr init-db
$ flask --app flaskr run --debug
```
    

Open http://127.0.0.1:5000 in a browser.


## Test
You'll need to install flaskr as a package:

    $ pip install '.[test]'
    $ pytest

Run with coverage report::

    $ coverage run -m pytest
    $ coverage report
    $ coverage html  # open htmlcov/index.html in a browser
