import json
import os


def load_json_file(path: str) -> dict:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected data file not found at '{path}'. "
            f"Make sure the 'data/' folder was copied along with the rest of the project."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"'{path}' exists but isn't valid JSON ({e}). Check for a stray comma or quote."
        )


def get_flower_info(flower_name: str, info_lookup: dict) -> dict | None:

    if not flower_name:
        return None

    normalized = flower_name.strip().upper()
    for key, value in info_lookup.items():
        if key.strip().upper() == normalized:
            return value
    return None


def format_confidence(confidence: float) -> str:
    """Turn a 0-1 confidence score into a display string like '98.76%'."""
    return f"{confidence * 100:.2f}%"


def get_api_key(key_name: str) -> str | None:

    try:
        import streamlit as st
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        # No secrets.toml present (normal when running locally) — fall through.
        pass

    from dotenv import load_dotenv
    load_dotenv()

    return os.environ.get(key_name)
