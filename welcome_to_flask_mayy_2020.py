# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 08:31:28 2020

@author: mhabayeb
"""

from flask import Flask
app = Flask(__name__)
@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!, Mayy"
if __name__ == '__main__':
 #   
    app.run(debug=True,port=12345)
