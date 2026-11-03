"""Step-wise Stage 2 preference training primitives."""

from .packing import (
    SidecarBranchRequest,
    SidecarExecutionPlan,
    build_sidecar_execution_plan,
)

__all__ = [
    "SidecarBranchRequest",
    "SidecarExecutionPlan",
    "build_sidecar_execution_plan",
]
