
from flask import Flask, jsonify
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

client = MongoClient(
    app.config["MONGO_URI"],
    serverSelectionTimeoutMS=3000
)

db = client[app.config["MONGO_DATABASE"]]


@app.route("/")
def home():
    return jsonify({
        "project": "Crisis Management System",
        "status": "running",
        "message": "Welcome to the Crisis Management System!"
    })


@app.route("/health")
def health():
    try:
        client.admin.command("ping")
        return jsonify({
            "application": "running",
            "database": "connected"
        }), 200
    except PyMongoError:
        app.logger.exception("MongoDB health check failed")
        return jsonify({
            "application": "running",
            "database": "disconnected"
        }), 503


if __name__ == "__main__":
    app.run(debug=True)
