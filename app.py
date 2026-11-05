
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from werkzeug.utils import secure_filename

from config import Config
from app.services.change_detection import predict_damage, save_heatmap


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
RESULT_DIR = BASE_DIR / "app" / "static" / "results"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB per request

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
app.config.from_object(Config)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

client = MongoClient(
    app.config["MONGO_URI"],
    serverSelectionTimeoutMS=3000,
)
db = client[app.config["MONGO_DATABASE"]]


def allowed_image(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        client.admin.command("ping")
        return jsonify({
            "application": "running",
            "database": "connected",
        }), 200
    except PyMongoError:
        app.logger.exception("MongoDB health check failed")
        return jsonify({
            "application": "running",
            "database": "disconnected",
        }), 503


@app.route("/analyze", methods=["POST"])
def analyze():
    pre_file = request.files.get("pre_image")
    post_file = request.files.get("post_image")

    if not pre_file or not post_file:
        return "Please upload both satellite images.", 400

    if not pre_file.filename or not post_file.filename:
        return "Please select both image files.", 400

    if not allowed_image(pre_file.filename) or not allowed_image(post_file.filename):
        return "Unsupported image format.", 400

    incident_id = uuid.uuid4().hex
    pre_name = f"{incident_id}_pre{Path(secure_filename(pre_file.filename)).suffix.lower()}"
    post_name = f"{incident_id}_post{Path(secure_filename(post_file.filename)).suffix.lower()}"
    heatmap_name = f"{incident_id}_heatmap.png"

    pre_path = UPLOAD_DIR / pre_name
    post_path = UPLOAD_DIR / post_name
    heatmap_path = RESULT_DIR / heatmap_name

    try:
        pre_file.save(pre_path)
        post_file.save(post_path)

        result = predict_damage(str(pre_path), str(post_path))
        save_heatmap(
            result["damage_map"],
            str(heatmap_path),
            str(post_path),
        )

        import numpy as np

        class_ids, counts = np.unique(
            result["damage_map"],
            return_counts=True,
        )

        record = {
            "incident_id": incident_id,
            "pre_image": pre_name,
            "post_image": post_name,
            "heatmap": heatmap_name,
            "image_size": result["image_size"],
            "building_pixels": int(result["building_map"].sum()),
            "predicted_class_pixel_counts": {
                str(int(class_id)): int(count)
                for class_id, count in zip(class_ids, counts)
            },
        }

        try:
            db.incidents.insert_one(record)
            database_status = "saved"
        except PyMongoError:
            app.logger.exception("Could not save incident to MongoDB")
            database_status = "not saved"

        return render_template(
            "result.html",
            record=record,
            pre_url=url_for("static", filename=f"uploads/{pre_name}"),
            post_url=url_for("static", filename=f"uploads/{post_name}"),
            heatmap_url=url_for("static", filename=f"results/{heatmap_name}"),
            database_status=database_status,
        )

    except Exception:
        app.logger.exception("Image analysis failed")
        return (
            "Image analysis failed. Check the Flask terminal for the error details.",
            500,
        )


@app.errorhandler(413)
def file_too_large(_error):
    return "Upload is too large. Please keep both images under the 20 MB request limit.", 413


if __name__ == "__main__":
    app.run(debug=True)
