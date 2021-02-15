from enum import Enum

# Numerical constants
POS_INF = 1e7
NEG_INF = -1e7
ZERO = 1e-7
ONE = 1 - ZERO
MIN_LIKELIHOOD = 1e-7

# UDS-related constants
class Type(Enum):
    EVENT = 0
    PARTICIPANT = 1
    ROLE = 2
    RELATION = 3


STR_TO_TYPE = {t.name: t for t in Type}

N_CONFIDENCE_SCORES = 5

# Even with new UDS metadata utilities, still need to manually
# identify which subspaces correspond to which nodes and edges
PREDICATE_NODE_SUBSPACES = {"time", "genericity", "factuality", "event_structure"}
ARGUMENT_NODE_SUBSPACES = {"genericity", "wordsense"}
SEMANTICS_EDGE_SUBSPACES = {"protoroles", "distributivity"}
DOCUMENT_EDGE_SUBSPACES = {"time", "mereology"}

EVENT_SUBSPACES = PREDICATE_NODE_SUBSPACES
PARTICIPANT_SUBSPACES = ARGUMENT_NODE_SUBSPACES
ROLE_SUBSPACES = SEMANTICS_EDGE_SUBSPACES
RELATION_SUBSPACES = DOCUMENT_EDGE_SUBSPACES

SUBSPACES_BY_TYPE = {
    Type.EVENT: EVENT_SUBSPACES,
    Type.PARTICIPANT: PARTICIPANT_SUBSPACES,
    Type.ROLE: ROLE_SUBSPACES,
    Type.RELATION: RELATION_SUBSPACES,
}

# Properties requiring an additional "does not apply" category
CONDITIONAL_PROPERTIES = {
    "situation_duration_lbound",
    "situation_duration_ubound",
    "dynamic",
    "part_similarity",
    "avg_part_duration_lbound",
    "avg_part_duration_ubound",
    "awareness",
    "change_of_location",
    "change_of_possession",
    "change_of_state",
    "change_of_state_continuous",
    "existed_after",
    "existed_before",
    "existed_during",
    "instigation",
    "location",
    "manner",
    "partitive",
    "purpose",
    "sentient",
    "time",
    "volition",
    "was_for_benefit",
    "was_used",
}
