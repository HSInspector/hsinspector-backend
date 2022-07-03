import os
import time
# import Test

from flask_cors import CORS #comment this on deployment
from flask import Flask, request, jsonify, json
from HSInspector import HSInspector
from flask_restful import Api, Resource, reqparse
app = Flask(__name__)
CORS(app) #comment this on deployment
api = Api(app)

app = Flask(__name__, static_url_path='', static_folder='frontend/build')
@app.route('/members', methods=['GET'])
def members():
    return {"members": ["Member1", "Member2", "Member2"]}

@app.route('/search/keyword', methods = ['POST'])
def create():
    request_data = json.loads(request.data)
    inspector = HSInspector()
    # print("OPTIONNN SELECTED")
    # print(request_data['option'])
    # js={"tweets":[], "types":[], "username":[]}
    if request_data['option']=='Keyword':
        # print("SEARCHEDDD BY KEYWORDDD")
        js = inspector.searchTweets(request_data['content'])
    elif request_data['option']=="Username":
        js = inspector.searchTweetsByUsername(request_data['content'])
    return js

if __name__ == "__main__":
    app.run(debug=True)