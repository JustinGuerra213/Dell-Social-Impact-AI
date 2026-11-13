"""Location-aware public-safety and historical flood research sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import requests

from flood_app.models import ImagePair


@dataclass(frozen=True)
class ResearchFinding:
    """One sourced research item for a report section."""

    title: str
    detail: str
    url: str
    observed_at: str = ""
    source_name: str = ""


@dataclass(frozen=True)
class AreaResearch:
    """Dynamic alerts and relevant official regional historical sources."""

    county: str
    checked_at: str
    active_alerts: tuple[ResearchFinding, ...]
    historical_disasters: tuple[ResearchFinding, ...]
    regional_reports: tuple[ResearchFinding, ...]
    errors: tuple[str, ...]


class AreaResearchService:
    """Fetch current official alerts plus curated area-relevant reports."""

    NWS_ALERTS = "https://api.weather.gov/alerts/active"
    FEMA_DECLARATIONS = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
    TIMEOUT = (5, 15)

    def research(self, pair: ImagePair) -> AreaResearch:
        """Build a research bundle using the image pair's georeferenced context."""
        record = pair.after
        county = record.county_name or "County not identified in image metadata"
        errors: list[str] = []
        alerts: tuple[ResearchFinding, ...] = ()
        disasters: tuple[ResearchFinding, ...] = ()
        if record.latitude is not None and record.longitude is not None:
            try:
                alerts = self._active_alerts(record.latitude, record.longitude)
            except (requests.RequestException, ValueError, KeyError) as exc:
                errors.append(f"NWS active alerts unavailable: {exc}")
        else:
            errors.append("No georeferenced coordinates; location-specific live alerts were not queried.")

        if record.county_name:
            try:
                disasters = self._historical_disasters(record.county_name)
            except (requests.RequestException, ValueError, KeyError) as exc:
                errors.append(f"FEMA disaster history unavailable: {exc}")

        return AreaResearch(
            county=county,
            checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            active_alerts=alerts,
            historical_disasters=disasters,
            regional_reports=self._regional_reports(pair),
            errors=tuple(errors),
        )

    def _active_alerts(self, latitude: float, longitude: float) -> tuple[ResearchFinding, ...]:
        response = requests.get(
            self.NWS_ALERTS,
            params={"point": f"{latitude},{longitude}"},
            headers={"User-Agent": "SavingTexas/1.0 (flood impact triage)", "Accept": "application/geo+json"},
            timeout=self.TIMEOUT,
        )
        response.raise_for_status()
        features = response.json().get("features", [])
        findings: list[ResearchFinding] = []
        for feature in features[:8]:
            properties = feature.get("properties", {})
            findings.append(
                ResearchFinding(
                    title=str(properties.get("event", "Active NWS alert")),
                    detail=str(properties.get("headline") or properties.get("description") or "See alert details."),
                    url=str(properties.get("web") or feature.get("id") or self.NWS_ALERTS),
                    observed_at=str(properties.get("sent", "")),
                    source_name="National Weather Service",
                )
            )
        return tuple(findings)

    def _historical_disasters(self, county: str) -> tuple[ResearchFinding, ...]:
        # OpenFEMA's disaster-declaration records name the county in designatedArea.
        county_name = county.removesuffix(" County")
        county_designation = f"{county_name} (County)"
        filter_value = f"state eq 'TX' and designatedArea eq '{county_designation}'"
        response = requests.get(
            self.FEMA_DECLARATIONS,
            params={
                "$filter": filter_value,
                "$orderby": "declarationDate desc",
                "$top": 12,
                "$select": "disasterNumber,declarationTitle,declarationDate,incidentType,incidentBeginDate,incidentEndDate",
            },
            headers={"User-Agent": "SavingTexas/1.0"},
            timeout=self.TIMEOUT,
        )
        response.raise_for_status()
        rows = response.json().get("DisasterDeclarationsSummaries", [])
        return tuple(
            ResearchFinding(
                title=f"{row.get('declarationTitle', 'FEMA disaster')} · {row.get('incidentType', 'incident')}",
                detail=(
                    f"FEMA declaration {row.get('disasterNumber')}; declared {row.get('declarationDate', 'date unavailable')}; "
                    f"incident dates {row.get('incidentBeginDate', '?')} to {row.get('incidentEndDate', '?')}. "
                    "This is a county-level declaration, not proof this image footprint was affected."
                ),
                url=(
                    "https://www.fema.gov/disaster/"
                    + str(row.get("disasterNumber", ""))
                ),
                observed_at=str(row.get("declarationDate", "")),
                source_name="FEMA OpenFEMA",
            )
            for row in rows
        )

    @staticmethod
    def _regional_reports(pair: ImagePair) -> tuple[ResearchFinding, ...]:
        event = pair.before.event_name.lower()
        locality = pair.before.locality.lower()
        looks_houston = any(term in locality for term in ("houston", "harris", "bear creek", "addicks", "barker"))
        looks_harvey = "harvey" in event
        if not looks_houston and not looks_harvey:
            return (
                ResearchFinding(
                    "Use county and state reports for local validation",
                    "No curated Houston-area historical report was selected for this location. County-level source results below are regional context only.",
                    "https://www.fema.gov/about/openfema/api",
                    source_name="FEMA OpenFEMA",
                ),
            )
        return (
            ResearchFinding(
                "Harris County drainage system context",
                "Harris County Flood Control District describes roughly 2,500 miles of channels and notes that extreme rainfall can exceed channel capacity. This is regional background, not a diagnosis of the selected site.",
                "https://www.hcfcd.org/About/Flooding-and-Floodplains/Drainage-Network",
                source_name="Harris County Flood Control District",
            ),
            ResearchFinding(
                "Harvey sediment and channel restoration benchmark",
                "HCFCD reported a FEMA award of nearly $250 million for sediment removal across eight watersheds, with an estimated 2.13 million cubic yards deposited during Harvey. This historical program amount is not a cost estimate for this image pair.",
                "https://www.hcfcd.org/Community/Communications/News-Room?post=The+Harris+County+Flood+Control+District+to+Receive+Nearly+%24250+Million+In+Federal+Funds+for+Sediment+Removal",
                observed_at="2021-08-09",
                source_name="Harris County Flood Control District / FEMA",
            ),
            ResearchFinding(
                "Harvey chemical-release and water-system reports",
                "EPA/TCEQ reported monitoring water and wastewater systems and investigating storm-related releases, including a refinery tank spill in Houston. Historical incidents do not establish current contamination at this site; current sampling and incident reports are required.",
                "https://www.epa.gov/archive/epa/newsreleases/epatceq-updated-status-systems-affected-harvey-2.html",
                source_name="EPA / TCEQ",
            ),
            ResearchFinding(
                "Evacuation zones and route planning",
                "Use the Ready Harris interactive evacuation map and active local instructions. This app does not infer a safe route from aerial images; road closures and water depth must be checked in real time.",
                "https://www.readyharris.org/Stay-Safe/Evacuation",
                source_name="Ready Harris",
            ),
        )
