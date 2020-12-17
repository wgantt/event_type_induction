from enum import Enum

# Numerical constants
POS_INF = 1e10
NEG_INF = -1e10
ZERO = 1e-10
ONE = 1 - ZERO
MIN_LIKELIHOOD = 1e-3

# UDS-related constants

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
