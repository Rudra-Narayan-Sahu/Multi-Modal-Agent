import datetime
from typing import Dict, Any, Optional
import pytz


class DateTimeUtil:
    """Utility for real-time datetime operations and timezone conversions."""

    @staticmethod
    def get_current_datetime(timezone_name: str = "UTC") -> Dict[str, Any]:
        """Get the current timestamp in UTC and an optional target timezone."""
        try:
            tz = pytz.timezone(timezone_name)
        except Exception:
            tz = pytz.UTC
            timezone_name = "UTC"

        now_tz = datetime.datetime.now(tz)
        now_utc = datetime.datetime.now(pytz.UTC)

        return {
            "success": True,
            "timezone": timezone_name,
            "iso": now_tz.isoformat(),
            "formatted": now_tz.strftime("%A, %B %d, %Y %I:%M:%S %p %Z"),
            "date": now_tz.strftime("%Y-%m-%d"),
            "time": now_tz.strftime("%H:%M:%S"),
            "day_of_week": now_tz.strftime("%A"),
            "utc_iso": now_utc.isoformat(),
        }

    @staticmethod
    def calculate_date_delta(days: int = 0, hours: int = 0, base_iso: Optional[str] = None) -> Dict[str, Any]:
        """Add or subtract days/hours from a base date (defaults to now)."""
        try:
            if base_iso:
                base = datetime.datetime.fromisoformat(base_iso)
            else:
                base = datetime.datetime.now(pytz.UTC)
            
            target = base + datetime.timedelta(days=days, hours=hours)
            return {
                "success": True,
                "base": base.isoformat(),
                "result": target.isoformat(),
                "formatted": target.strftime("%A, %B %d, %Y %I:%M:%S %p UTC")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

