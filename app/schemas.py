from enum import Enum
from typing import Optional
import uuid
from pydantic import BaseModel, Field
try:
    from pydantic import field_validator
    def validator_compat(*fields, **kwargs):
        return field_validator(*fields, mode='after')
except ImportError:
    from pydantic import validator as validator_compat

class ProfileEnum(str, Enum):
    interactive = "interactive"
    normal = "normal"
    batch = "batch"

class PolicyEnum(str, Enum):
    fifo = "fifo"
    token_aware = "token_aware"
    adaptive = "adaptive"
    adaptive_least_conn = "adaptive_least_conn"

class InferenceRequest(BaseModel):
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(..., min_length=1)
    profile: ProfileEnum = Field(default=ProfileEnum.normal)
    max_output_tokens: int = Field(default=128, gt=0, le=1000)
    policy: PolicyEnum = Field(default=PolicyEnum.adaptive)
    simulate_failure: Optional[str] = None  # passed to backend if needed
    sim_true_output_tokens: Optional[int] = Field(default=None, gt=0, le=1000)
    use_real_model: bool = Field(default=False)

    @validator_compat('simulate_failure')
    def validate_failure(cls, v):
        if v is None:
            return v
        allowed = {"none", "before", "partial"}
        if v not in allowed:
            raise ValueError(f"simulate_failure must be one of {allowed}")
        return v
