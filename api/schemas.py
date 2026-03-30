"""Pydantic models for request validation and response serialisation.

AnalyzeRequest validates the form fields that accompany the uploaded image.
AnalyzeResponse carries the weight estimate and all supporting metadata back
to the caller.  ``input_height_cm`` is a passthrough of the user-supplied
value — it is never estimated from the image.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Displayed by the frontend whenever AnalyzeResponse.low_confidence is True.
LowConfidenceWarning: str = (
    "This result has low confidence. Try retaking the photo with your full "
    "body visible and facing the camera directly."
)


class AnalyzeRequest(BaseModel):
    """Structured validation for the non-image form fields in POST /analyze."""

    height_cm: float = Field(
        ...,
        ge=50.0,
        le=300.0,
        description="Standing height in centimetres (50–300).",
    )
    age: int = Field(
        ...,
        ge=5,
        le=120,
        description="Age in years (5–120).",
    )
    gender: int = Field(
        ...,
        ge=0,
        le=1,
        description="Biological sex: 0 for female, 1 for male.",
    )


class AnalyzeResponse(BaseModel):
    """Full response returned by POST /analyze."""

    estimated_weight_kg: float = Field(
        ..., description="Mean of all Monte Carlo forward pass outputs, in kg."
    )
    confidence_interval_low: float = Field(
        ..., description="Lower bound of the 95% confidence interval (mean − 1.96 × std)."
    )
    confidence_interval_high: float = Field(
        ..., description="Upper bound of the 95% confidence interval (mean + 1.96 × std)."
    )
    prediction_std: float = Field(
        ...,
        description=(
            "Standard deviation of the MC pass outputs. A high value indicates "
            "epistemic uncertainty — the input is out of the training distribution."
        ),
    )
    low_confidence: bool = Field(
        ...,
        description=(
            "True when prediction_std exceeds the configured threshold. "
            "The frontend should display LowConfidenceWarning when this is True."
        ),
    )
    input_height_cm: float = Field(
        ...,
        description="Passthrough of the user-supplied height_cm. Never estimated from the image.",
    )
    processing_time_ms: float = Field(
        ..., description="Wall-clock time from image decode to WeightEstimationResult, in ms."
    )
