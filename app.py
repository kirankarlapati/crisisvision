
import json
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for, send_file
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from werkzeug.utils import secure_filename

from config import Config
from app.services.change_detection import predict_damage, save_heatmap
from app.services.building_extraction import (
    extract_buildings,
    save_annotated_buildings_image,
)


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
    annotated_name = f"{incident_id}_annotated.png"
    json_name = f"{incident_id}_buildings.json"

    pre_path = UPLOAD_DIR / pre_name
    post_path = UPLOAD_DIR / post_name
    heatmap_path = RESULT_DIR / heatmap_name
    annotated_path = RESULT_DIR / annotated_name
    json_path = RESULT_DIR / json_name

    try:
        pre_file.save(pre_path)
        post_file.save(post_path)

        result = predict_damage(str(pre_path), str(post_path))
        save_heatmap(
            result["damage_map"],
            str(heatmap_path),
            str(post_path),
        )

        # Extract individual building footprints and damage metrics
        buildings_data = extract_buildings(
            result["building_map"],
            result["damage_map"],
            analysis_id=incident_id,
            min_area=25,
        )

        # Generate annotated image with bounding boxes & centroids
        save_annotated_buildings_image(
            str(heatmap_path),
            buildings_data,
            str(annotated_path),
        )

        # Persist structured building JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(buildings_data, f, indent=2)

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
            "annotated_image": annotated_name,
            "buildings_json": json_name,
            "image_size": result["image_size"],
            "building_pixels": int(result["building_map"].sum()),
            "total_buildings": buildings_data["total_buildings"],
            "damage_summary": buildings_data["damage_summary"],
            "predicted_class_pixel_counts": {
                str(int(class_id)): int(count)
                for class_id, count in zip(class_ids, counts)
            },
        }

        try:
            # We copy record and insert without modifying original dict
            db_record = dict(record)
            db.incidents.insert_one(db_record)
            database_status = "saved"
        except PyMongoError:
            app.logger.exception("Could not save incident to MongoDB")
            database_status = "not saved"

        return render_template(
            "result.html",
            record=record,
            buildings_data=buildings_data,
            pre_url=url_for("static", filename=f"uploads/{pre_name}"),
            post_url=url_for("static", filename=f"uploads/{post_name}"),
            heatmap_url=url_for("static", filename=f"results/{heatmap_name}"),
            annotated_url=url_for("static", filename=f"results/{annotated_name}"),
            json_url=url_for("static", filename=f"results/{json_name}"),
            database_status=database_status,
        )

    except Exception:
        app.logger.exception("Image analysis failed")
        return (
            "Image analysis failed. Check the Flask terminal for the error details.",
            500,
        )


@app.route("/api/incidents/<incident_id>/buildings")
def get_incident_buildings(incident_id):
    """Retrieve structured building-level JSON for a given incident."""
    safe_id = secure_filename(incident_id)
    json_path = RESULT_DIR / f"{safe_id}_buildings.json"
    if not json_path.exists():
        return jsonify({"error": "Incident not found"}), 404
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data), 200


@app.errorhandler(413)
def file_too_large(_error):
    return "Upload is too large. Please keep both images under the 20 MB request limit.", 413


if __name__ == "__main__":
    app.run(debug=True)
