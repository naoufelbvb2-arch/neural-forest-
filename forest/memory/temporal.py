"""TemporalService: injects the current date/time into system prompts."""

from __future__ import annotations

import datetime


class TemporalService:
    """Reads the system clock and formats current date/time for the model.

    This is an external service — it does NOT use neural weights. The date
    string is injected into the system prompt before each forward pass.

    Example output: "Today is Sunday, 25 May 2026. Current time: 14:30 UTC."

    TODO: implement in PROMPT 2
    """

    def get_datetime_string(self, tz: datetime.timezone = datetime.timezone.utc) -> str:
        """Return a human-readable date/time string.

        Args:
            tz: Timezone to use (default UTC).

        Returns:
            Formatted string for system prompt injection.

        TODO: implement in PROMPT 2
        """
        raise NotImplementedError("TemporalService.get_datetime_string — implement in PROMPT 2")
