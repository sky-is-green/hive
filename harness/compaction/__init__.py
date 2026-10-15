"""Conversation compaction and the engine-surgery offer at a checkpoint.

See :mod:`harness.compaction.core` for the function and :mod:`harness.compaction.surgery` for the
hand-off: a checkpoint is the transaction boundary where changing what serves a session is offered,
gated by tests, and rolled back to the checkpoint when the tests say no.
"""

from harness.compaction.core import (  # noqa: F401
    CHARS_PER_TOKEN,
    SUMMARY_HEADINGS,
    Checkpoint,
    Editor,
    EventSink,
    Message,
    Outcome,
    Settings,
    SummaryRequest,
    Summarizer,
    Trigger,
    build_prompt,
    ceiling,
    compact,
    default_counter,
    due,
    estimate,
    split,
    uses_template,
)
from harness.compaction.surgery import (  # noqa: F401
    Candidate,
    GateResult,
    Probe,
    SurgicalOutcome,
    offer_surgery,
)
