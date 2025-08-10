from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class User:
    """
    Extensible user model containing all user-related information.
    """
    id: str
    api_key: str
    model: str = "gpt-4o"  # Default AI model

    # User preferences
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "advanced_parsing": True,
        "batch_operations": True,
        "simplify_integration": True,
        "custom_templates": True,
    })

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary representation."""
        return {
            "id": self.id,
            "api_key": self.api_key[:10] + "..." if self.api_key else None,  # Masked
            "model": self.model,
            "preferences": self.preferences,
            "features": self.features,
            "quotas": {
                "max_jobs": self.max_jobs,
                "max_resumes": self.max_resumes,
                "max_api_calls_per_day": self.max_api_calls_per_day
            },
            "metadata": self.metadata
        }

    def has_feature(self, feature_name: str) -> bool:
        """Check if user has access to a specific feature."""
        return self.features.get(feature_name, False)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference with optional default."""
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.preferences[key] = value