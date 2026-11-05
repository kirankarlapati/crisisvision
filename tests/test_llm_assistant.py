"""
Unit and Integration Tests for CrisisVision LLM Assistant Service.

Verifies:
  - Incident context building and schema validation.
  - Damage and hotspot summary aggregation.
  - Automatic explanation generation (including zero-damage deterministic case).
  - Chat question answering and conversation history handling.
  - Ollama error handling (unreachable, timeout, empty response).
  - Flask endpoints (/api/incidents/<id>/explanation and /api/incidents/<id>/ask).
"""

import json
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("root_app", str(ROOT_DIR / "app.py"))
root_app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_app_mod)
flask_app = root_app_mod.app
RESULT_DIR = root_app_mod.RESULT_DIR

from app.services.llm_assistant import (
    SYSTEM_PROMPT,
    STANDARD_LIMITATIONS,
    LLMContextError,
    LLMServiceError,
    build_incident_context,
    validate_incident_context,
    generate_damage_explanation,
    answer_incident_question,
)


@pytest.fixture
def sample_buildings_data():
    return {
        "analysis_id": "test_inc_001",
        "coordinate_system": "image_pixel",
        "image_width": 1024,
        "image_height": 1024,
        "total_buildings": 4,
        "damage_summary": {
            "no_damage": 1,
            "minor_damage": 1,
            "major_damage": 1,
            "destroyed": 1,
        },
        "buildings": [
            {
                "building_id": "B001",
                "centroid": {"x": 100.0, "y": 100.0},
                "bounding_box": {"x_min": 90, "y_min": 90, "x_max": 110, "y_max": 110},
                "area_pixels": 200,
                "damage_class_id": 1,
                "damage_label": "no_damage",
                "confidence": 0.95,
            },
            {
                "building_id": "B002",
                "centroid": {"x": 200.0, "y": 200.0},
                "bounding_box": {"x_min": 190, "y_min": 190, "x_max": 210, "y_max": 210},
                "area_pixels": 250,
                "damage_class_id": 2,
                "damage_label": "minor_damage",
                "confidence": 0.88,
            },
            {
                "building_id": "B003",
                "centroid": {"x": 300.0, "y": 300.0},
                "bounding_box": {"x_min": 290, "y_min": 290, "x_max": 310, "y_max": 310},
                "area_pixels": 300,
                "damage_class_id": 3,
                "damage_label": "major_damage",
                "confidence": 0.85,
            },
            {
                "building_id": "B004",
                "centroid": {"x": 500.0, "y": 500.0},
                "bounding_box": {"x_min": 480, "y_min": 480, "x_max": 520, "y_max": 520},
                "area_pixels": 400,
                "damage_class_id": 4,
                "damage_label": "destroyed",
                "confidence": 0.92,
            },
        ],
    }


@pytest.fixture
def sample_hotspots_data():
    return {
        "analysis_id": "test_inc_001",
        "coordinate_system": "image_pixel",
        "image_width": 1024,
        "image_height": 1024,
        "summary": {
            "total_hotspots": 2,
            "critical_priority_hotspots": 1,
            "high_priority_hotspots": 1,
            "moderate_priority_hotspots": 0,
            "low_priority_hotspots": 0,
        },
        "hotspots": [
            {
                "hotspot_id": "H001",
                "priority_category": "critical_priority",
                "priority_score": 0.85,
                "building_count": 2,
                "minor_damage_count": 0,
                "major_damage_count": 1,
                "destroyed_count": 1,
                "centroid": {"x": 490.5, "y": 495.2},
                "bounding_box": {"x_min": 450, "y_min": 450, "x_max": 540, "y_max": 540},
                "building_ids": ["B003", "B004"],
            },
            {
                "hotspot_id": "H002",
                "priority_category": "high_priority",
                "priority_score": 0.62,
                "building_count": 1,
                "minor_damage_count": 1,
                "major_damage_count": 0,
                "destroyed_count": 0,
                "centroid": {"x": 200.0, "y": 200.0},
                "bounding_box": {"x_min": 180, "y_min": 180, "x_max": 220, "y_max": 220},
                "building_ids": ["B002"],
            },
        ],
    }


@pytest.fixture
def zero_damage_data():
    buildings = {
        "analysis_id": "zero_inc_002",
        "coordinate_system": "image_pixel",
        "image_width": 1024,
        "image_height": 1024,
        "total_buildings": 5,
        "damage_summary": {
            "no_damage": 5,
            "minor_damage": 0,
            "major_damage": 0,
            "destroyed": 0,
        },
        "buildings": [
            {
                "building_id": f"B{i:03d}",
                "centroid": {"x": float(i * 100), "y": float(i * 100)},
                "bounding_box": {"x_min": i * 100, "y_min": i * 100, "x_max": i * 100 + 20, "y_max": i * 100 + 20},
                "area_pixels": 200,
                "damage_class_id": 1,
                "damage_label": "no_damage",
                "confidence": 1.0,
            }
            for i in range(1, 6)
        ],
    }
    hotspots = {
        "analysis_id": "zero_inc_002",
        "coordinate_system": "image_pixel",
        "image_width": 1024,
        "image_height": 1024,
        "summary": {
            "total_hotspots": 0,
            "critical_priority_hotspots": 0,
            "high_priority_hotspots": 0,
            "moderate_priority_hotspots": 0,
            "low_priority_hotspots": 0,
        },
        "hotspots": [],
    }
    return buildings, hotspots


class TestContextConstruction:
    def test_build_incident_context_valid(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)

        assert ctx["incident_id"] == "test_inc_001"
        assert ctx["coordinate_system"] == "image_pixel"
        assert ctx["image_dimensions"] == {"width": 1024, "height": 1024}

        # Building damage summary
        bs = ctx["building_summary"]
        assert bs["total_buildings"] == 4
        assert bs["damaged_buildings_total"] == 3
        assert bs["no_damage"] == 1
        assert bs["minor_damage"] == 1
        assert bs["major_damage"] == 1
        assert bs["destroyed"] == 1

        # Hotspot summary
        hs = ctx["hotspot_summary"]
        assert hs["total_hotspots"] == 2
        assert hs["critical_priority_hotspots"] == 1
        assert hs["high_priority_hotspots"] == 1

        # Hotspot details
        assert len(ctx["hotspots"]) == 2
        h0 = ctx["hotspots"][0]
        assert h0["hotspot_id"] == "H001"
        assert h0["priority_category"] == "critical_priority"
        assert h0["priority_score"] == 85.0
        assert h0["building_ids"] == ["B003", "B004"]
        assert h0["centroid"] == {"x": 490.5, "y": 495.2}

        # Limitations
        assert len(ctx["limitations"]) == len(STANDARD_LIMITATIONS)
        assert any("image pixels" in lim.lower() for lim in ctx["limitations"])

    def test_invalid_input_types_raise(self):
        with pytest.raises(LLMContextError, match="buildings_data must be a dictionary"):
            build_incident_context(None, {})  # type: ignore

        with pytest.raises(LLMContextError, match="hotspots_data must be a dictionary"):
            build_incident_context({}, "invalid")  # type: ignore

    def test_validation_missing_key_raises(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)
        del ctx["building_summary"]
        with pytest.raises(LLMContextError, match="Missing required context key"):
            validate_incident_context(ctx)

    def test_validation_invalid_coordinate_system_raises(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)
        ctx["coordinate_system"] = "gps_wgs84"
        with pytest.raises(LLMContextError, match="Invalid coordinate_system"):
            validate_incident_context(ctx)


class TestAutomaticExplanation:
    def test_zero_damage_produces_deterministic_explanation(self, zero_damage_data):
        buildings, hotspots = zero_damage_data
        ctx = build_incident_context(buildings, hotspots)

        res = generate_damage_explanation(ctx)
        assert res["success"] is True
        assert res["incident_id"] == "zero_inc_002"
        assert "no detected damage" in res["explanation"].lower()
        assert "5 building candidates" in res["explanation"]
        assert "no damage hotspots" in res["explanation"].lower()

    def test_mocked_ollama_explanation_success(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "content": (
                    "Executive Assessment for Incident test_inc_001:\n"
                    "4 buildings were extracted, of which 3 suffered damage (1 destroyed, 1 major, 1 minor).\n"
                    "2 hotspots were identified, with H001 classified as critical priority (score: 85%).\n"
                    "Locations are in pixel coordinates."
                )
            }
        }

        res = generate_damage_explanation(ctx, client=mock_client)
        assert res["success"] is True
        assert res["incident_id"] == "test_inc_001"
        assert "4 buildings were extracted" in res["explanation"]
        assert mock_client.chat.called

        # Verify system prompt and messages were passed correctly
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "llama3.2"
        assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT

    def test_ollama_unavailable_returns_friendly_error(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)

        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionRefusedError("[WinError 10061] No connection could be made")

        res = generate_damage_explanation(ctx, client=mock_client)
        assert res["success"] is False
        assert res["explanation"] is None
        assert "ollama serve" in res["error"].lower()

    def test_ollama_empty_response_handled(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)

        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": ""}}

        res = generate_damage_explanation(ctx, client=mock_client)
        assert res["success"] is False
        assert "error" in res


class TestIncidentChat:
    def test_empty_question_rejected(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)
        res = answer_incident_question(ctx, question="   ")
        assert res["success"] is False
        assert "Question cannot be empty" in res["error"]

    def test_chat_question_success(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {
                "content": "Hotspot H001 has the highest priority score at 85.0%, containing 2 buildings."
            }
        }

        res = answer_incident_question(
            ctx,
            question="Which hotspot is the worst?",
            chat_history=[{"role": "user", "content": "Hello"}],
            client=mock_client,
        )

        assert res["success"] is True
        assert res["question"] == "Which hotspot is the worst?"
        assert "H001" in res["answer"]

        # Check call arguments
        call_kwargs = mock_client.chat.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert "test_inc_001" in messages[1]["content"]

    def test_chat_error_handling(self, sample_buildings_data, sample_hotspots_data):
        ctx = build_incident_context(sample_buildings_data, sample_hotspots_data)

        mock_client = MagicMock()
        mock_client.chat.side_effect = TimeoutError("Request timed out")

        res = answer_incident_question(ctx, question="How many buildings?", client=mock_client)
        assert res["success"] is False
        assert "timed out" in res["error"].lower()


class TestFlaskEndpoints:
    @pytest.fixture
    def test_client(self, tmp_path):
        flask_app.config["TESTING"] = True
        return flask_app.test_client()

    def test_ask_endpoint_missing_incident_404(self, test_client):
        resp = test_client.post("/api/incidents/nonexistent_12345/ask", json={"question": "Summarize"})
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_ask_endpoint_empty_question_400(self, test_client, tmp_path, sample_buildings_data, sample_hotspots_data):
        inc_id = "test_flask_001"
        b_file = RESULT_DIR / f"{inc_id}_buildings.json"
        h_file = RESULT_DIR / f"{inc_id}_hotspots.json"

        with open(b_file, "w", encoding="utf-8") as f:
            json.dump(sample_buildings_data, f)
        with open(h_file, "w", encoding="utf-8") as f:
            json.dump(sample_hotspots_data, f)

        try:
            resp = test_client.post(f"/api/incidents/{inc_id}/ask", json={"question": "   "})
            assert resp.status_code == 400
            data = resp.get_json()
            assert data["success"] is False
            assert "empty" in data["error"].lower()
        finally:
            if b_file.exists():
                b_file.unlink()
            if h_file.exists():
                h_file.unlink()

    @patch("app.services.llm_assistant.get_ollama_client")
    def test_ask_endpoint_success(
        self, mock_get_client, test_client, sample_buildings_data, sample_hotspots_data
    ):
        inc_id = "test_flask_002"
        b_file = RESULT_DIR / f"{inc_id}_buildings.json"
        h_file = RESULT_DIR / f"{inc_id}_hotspots.json"

        with open(b_file, "w", encoding="utf-8") as f:
            json.dump(sample_buildings_data, f)
        with open(h_file, "w", encoding="utf-8") as f:
            json.dump(sample_hotspots_data, f)

        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "There is 1 destroyed building in this incident."}
        }
        mock_get_client.return_value = mock_client

        try:
            resp = test_client.post(
                f"/api/incidents/{inc_id}/ask",
                json={"question": "How many destroyed buildings?"},
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["success"] is True
            assert "1 destroyed building" in data["answer"]
        finally:
            if b_file.exists():
                b_file.unlink()
            if h_file.exists():
                h_file.unlink()

    def test_explanation_endpoint_zero_damage(
        self, test_client, zero_damage_data
    ):
        inc_id = "test_flask_zero"
        b_data, h_data = zero_damage_data
        b_file = RESULT_DIR / f"{inc_id}_buildings.json"
        h_file = RESULT_DIR / f"{inc_id}_hotspots.json"
        exp_file = RESULT_DIR / f"{inc_id}_explanation.json"

        with open(b_file, "w", encoding="utf-8") as f:
            json.dump(b_data, f)
        with open(h_file, "w", encoding="utf-8") as f:
            json.dump(h_data, f)

        try:
            resp = test_client.get(f"/api/incidents/{inc_id}/explanation")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["success"] is True
            assert "no detected damage" in data["explanation"].lower()
            # File should be cached
            assert exp_file.exists()
        finally:
            for p in [b_file, h_file, exp_file]:
                if p.exists():
                    p.unlink()
