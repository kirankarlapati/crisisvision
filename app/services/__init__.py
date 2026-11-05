"""
CrisisVision Services Package.
"""

from app.services.change_detection import predict_damage, save_heatmap
from app.services.building_extraction import (
    extract_buildings,
    validate_building_data,
    save_annotated_buildings_image,
    DAMAGE_LABELS,
    DAMAGE_DISPLAY_NAMES,
)
from app.services.risk_mapping import (
    calculate_building_severity,
    generate_damage_heatmap,
    extract_hotspots,
    validate_risk_data,
    save_risk_map_image,
    DEFAULT_PRIORITY_THRESHOLDS,
    PRIORITY_DISPLAY_NAMES,
    PRIORITY_COLORS,
)
from app.services.llm_assistant import (
    build_incident_context,
    validate_incident_context,
    generate_damage_explanation,
    answer_incident_question,
    get_ollama_client,
    SYSTEM_PROMPT,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_HOST,
    LLMContextError,
    LLMServiceError,
)

__all__ = [
    "predict_damage",
    "save_heatmap",
    "extract_buildings",
    "validate_building_data",
    "save_annotated_buildings_image",
    "DAMAGE_LABELS",
    "DAMAGE_DISPLAY_NAMES",
    "calculate_building_severity",
    "generate_damage_heatmap",
    "extract_hotspots",
    "validate_risk_data",
    "save_risk_map_image",
    "DEFAULT_PRIORITY_THRESHOLDS",
    "PRIORITY_DISPLAY_NAMES",
    "PRIORITY_COLORS",
    "build_incident_context",
    "validate_incident_context",
    "generate_damage_explanation",
    "answer_incident_question",
    "get_ollama_client",
    "SYSTEM_PROMPT",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_HOST",
    "LLMContextError",
    "LLMServiceError",
]
