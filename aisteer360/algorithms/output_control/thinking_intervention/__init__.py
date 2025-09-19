from .args import ThinkingInterventionArgs
from .control import ThinkingIntervention

# __all__ = ["ThinkingIntervention", "ThinkingInterventionArgs"]

STEERING_METHOD = {
    "category": "output_control",
    "name": "thinking_intervention",
    "control": ThinkingIntervention,
    "args": ThinkingInterventionArgs,
}
