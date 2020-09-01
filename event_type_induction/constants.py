from enum import Enum

# Numerical constants
POS_INF = 1e10
NEG_INF = -1e10
ZERO = 1e-10
ONE = 1 - ZERO

# UDS-related constants

# Data types (currently treating duration as nominal instead
# of ordinal, so we don't have any ordinal variables at the moment)
ORDINAL = 0
NOMINAL = 1
BINARY = 2

BINARY_PROP_SIZE = 1
N_CONFIDENCE_SCORES = 5


PREDICATE_ANNOTATION_ATTRIBUTES = {
    "time": {"duration": {"type": NOMINAL, "dim": 11}},
    "genericity": {
        "pred-dynamic": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "pred-hypothetical": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "pred-particular": {"type": BINARY, "dim": BINARY_PROP_SIZE},
    },
    "factuality": {"factual": {"type": BINARY, "dim": BINARY_PROP_SIZE}},
    "event_structure": {
        "natural_parts": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "telic": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "situation_duration_lbound": {"type": NOMINAL, "dim": 12},
        "situation_duration_ubound": {"type": NOMINAL, "dim": 12},
        "avg_part_duration_lbound": {"type": NOMINAL, "dim": 12},
        "avg_part_duration_ubound": {"type": NOMINAL, "dim": 12},
        "part_similarity": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "dynamic": {"type": BINARY, "dim": BINARY_PROP_SIZE},
    },
}

ARGUMENT_ANNOTATION_ATTRIBUTES = {
    "genericity": {
        "arg-abstract": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "arg-kind": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "arg-particular": {"type": BINARY, "dim": BINARY_PROP_SIZE},
    },
    "wordsense": {
        "supersense-noun.shape": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.process": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.relation": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.communication": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.time": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.plant": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.phenomenon": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.animal": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.state": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.substance": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.person": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.possession": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.Tops": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.object": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.event": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.artifact": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.act": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.body": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.attribute": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.quantity": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.motive": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.location": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.cognition": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.group": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.food": {"type": BINARY, "dim": BINARY_PROP_SIZE},
        "supersense-noun.feeling": {"type": BINARY, "dim": BINARY_PROP_SIZE},
    },
}

SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES = {
    "protoroles": {
        "was_used": {"type": NOMINAL, "dim": 6},
        "purpose": {"type": NOMINAL, "dim": 6},
        "partitive": {"type": NOMINAL, "dim": 6},
        "location": {"type": NOMINAL, "dim": 6},
        "instigation": {"type": NOMINAL, "dim": 6},
        "existed_after": {"type": NOMINAL, "dim": 6},
        "time": {"type": NOMINAL, "dim": 6},
        "awareness": {"type": NOMINAL, "dim": 6},
        "change_of_location": {"type": NOMINAL, "dim": 6},
        "manner": {"type": NOMINAL, "dim": 6},
        "sentient": {"type": NOMINAL, "dim": 6},
        "was_for_benefit": {"type": NOMINAL, "dim": 6},
        "change_of_state_continuous": {"type": NOMINAL, "dim": 6,},
        "existed_during": {"type": NOMINAL, "dim": 6},
        "change_of_possession": {"type": NOMINAL, "dim": 6},
        "existed_before": {"type": NOMINAL, "dim": 6},
        "volition": {"type": NOMINAL, "dim": 6},
        "change_of_state": {"type": NOMINAL, "dim": 6},
    },
    "distributivity": {"distributive": {"type": BINARY, "dim": BINARY_PROP_SIZE}},
}

DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES = {
    "mereology": {
        "containment.p1_contains_p2": {"type": BINARY, "dim": BINARY_PROP_SIZE,},
        "containment.p2_contains_p1": {"type": BINARY, "dim": BINARY_PROP_SIZE,},
    },
    "time": {"temporal-relation": {"type": NOMINAL, "dim": 4}},
}
