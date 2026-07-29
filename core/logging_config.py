"""JSON logging with Elastic Common Schema-inspired fields."""

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone

request_id_var = contextvars.ContextVar("request_id", default=None)

_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Emit one structured JSON event per line for Docker and Elastic ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "log.level": record.levelname.lower(),
            "message": record.getMessage(),
            "log.logger": record.name,
            "service.name": "jobagent-api",
        }
        request_id = request_id_var.get()
        if request_id:
            event["trace.id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                event[key] = value
        if record.exc_info:
            event["error.stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(event, default=str, ensure_ascii=False)


def configure_logging() -> None:
    """Configure stdout-only structured logs; Docker owns collection and rotation."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
