
import os


class Config:
    MONGO_URI = os.getenv(
        "MONGO_URI",
        "mongodb://127.0.0.1:27017/"
    )

    MONGO_DATABASE = os.getenv(
        "MONGO_DATABASE",
        "crisis_management"
    )

    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "dev-only-change-this-before-deployment"
    )
