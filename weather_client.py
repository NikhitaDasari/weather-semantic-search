"""National Weather Service API client for forecasts and active alerts."""

import hashlib
import os
from typing import Any

import requests

NWS_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
DEFAULT_TIMEOUT = 30


class WeatherClient:
    def __init__(self, base_url: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = (base_url or NWS_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "weather-semantic-search/1.0",
            "Accept": "application/geo+json",
        })

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def get_forecast(self, latitude: float, longitude: float) -> dict:
        point = self.get(f"/points/{latitude},{longitude}")
        forecast_url = point["properties"]["forecast"]
        response = self.session.get(forecast_url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_alerts(self, latitude: float, longitude: float) -> list[dict]:
        data = self.get(
            "/alerts/active", params={"point": f"{latitude},{longitude}"}
        )
        return data.get("features", [])

    def get_weather_documents(
        self,
        location_name: str,
        latitude: float,
        longitude: float,
        limit: int = 50,
    ) -> list[dict]:
        """Normalize NWS forecasts and alerts into a shared document schema."""
        documents: list[dict] = []

        forecast = self.get_forecast(latitude, longitude)
        properties = forecast.get("properties", {})
        for period in properties.get("periods", []):
            narrative = (period.get("detailedForecast") or "").strip()
            if not narrative:
                continue
            start_time = period.get("startTime")
            raw_id = f"{location_name}|forecast|{period.get('number')}|{start_time}"
            documents.append({
                "id": hashlib.sha256(raw_id.encode()).hexdigest(),
                "location": location_name,
                "source_type": "forecast",
                "headline": period.get("name"),
                "narrative_text": narrative,
                "issued_at": properties.get("generatedAt"),
                "effective_at": start_time,
                "payload": period,
            })

        for alert in self.get_alerts(latitude, longitude):
            properties = alert.get("properties", {})
            narrative = "\n\n".join(
                value.strip()
                for value in [
                    properties.get("description") or "",
                    properties.get("instruction") or "",
                ]
                if value.strip()
            )
            if not narrative:
                continue
            alert_id = alert.get("id") or properties.get("id")
            if not alert_id:
                raw_id = (
                    f"{location_name}|alert|{properties.get('event')}|"
                    f"{properties.get('sent')}"
                )
                alert_id = hashlib.sha256(raw_id.encode()).hexdigest()
            documents.append({
                "id": str(alert_id),
                "location": location_name,
                "source_type": "alert",
                "headline": properties.get("event"),
                "narrative_text": narrative,
                "issued_at": properties.get("sent"),
                "effective_at": properties.get("effective"),
                "payload": alert,
            })

        return documents[:limit]
