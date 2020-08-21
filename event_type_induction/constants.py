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

PREDICATE_ANNOTATION_ATTRIBUTES = {
    "time": {"duration": {"type": NOMINAL, "dim": 11}},
    "genericity": {
        "pred-dynamic": {"type": BINARY, "dim": 1},
        "pred-hypothetical": {"type": BINARY, "dim": 1},
        "pred-particular": {"type": BINARY, "dim": 1},
    },
    "factuality": {"factual": {"type": BINARY, "dim": 1}},
    "event_structure": {
        "natural_parts": {"type": BINARY, "dim": 1},
        "telic": {"type": BINARY, "dim": 1},
        "situation_duration_lbound": {"type": NOMINAL, "dim": 12},
        "situation_duration_ubound": {"type": NOMINAL, "dim": 12},
        "avg_part_duration_lbound": {"type": NOMINAL, "dim": 12},
        "avg_part_duration_ubound": {"type": NOMINAL, "dim": 12},
        "part_similarity": {"type": BINARY, "dim": 1},
        "dynamic": {"type": BINARY, "dim": 1},
    },
}

ARGUMENT_ANNOTATION_ATTRIBUTES = {
    "genericity": {
        "arg-abstract": {"type": BINARY, "dim": 1},
        "arg-kind": {"type": BINARY, "dim": 1},
        "arg-particular": {"type": BINARY, "dim": 1},
    },
    "wordsense": {
        "supersense-noun.shape": {"type": BINARY, "dim": 1},
        "supersense-noun.process": {"type": BINARY, "dim": 1},
        "supersense-noun.relation": {"type": BINARY, "dim": 1},
        "supersense-noun.communication": {"type": BINARY, "dim": 1},
        "supersense-noun.time": {"type": BINARY, "dim": 1},
        "supersense-noun.plant": {"type": BINARY, "dim": 1},
        "supersense-noun.phenomenon": {"type": BINARY, "dim": 1},
        "supersense-noun.animal": {"type": BINARY, "dim": 1},
        "supersense-noun.state": {"type": BINARY, "dim": 1},
        "supersense-noun.substance": {"type": BINARY, "dim": 1},
        "supersense-noun.person": {"type": BINARY, "dim": 1},
        "supersense-noun.possession": {"type": BINARY, "dim": 1},
        "supersense-noun.Tops": {"type": BINARY, "dim": 1},
        "supersense-noun.object": {"type": BINARY, "dim": 1},
        "supersense-noun.event": {"type": BINARY, "dim": 1},
        "supersense-noun.artifact": {"type": BINARY, "dim": 1},
        "supersense-noun.act": {"type": BINARY, "dim": 1},
        "supersense-noun.body": {"type": BINARY, "dim": 1},
        "supersense-noun.attribute": {"type": BINARY, "dim": 1},
        "supersense-noun.quantity": {"type": BINARY, "dim": 1},
        "supersense-noun.motive": {"type": BINARY, "dim": 1},
        "supersense-noun.location": {"type": BINARY, "dim": 1},
        "supersense-noun.cognition": {"type": BINARY, "dim": 1},
        "supersense-noun.group": {"type": BINARY, "dim": 1},
        "supersense-noun.food": {"type": BINARY, "dim": 1},
        "supersense-noun.feeling": {"type": BINARY, "dim": 1},
    },
}

SEMANTICS_EDGE_ANNOTATION_ATTRIBUTES = {
    "protoroles": {
        "was_used": {"type": BINARY, "dim": 1},
        "purpose": {"type": BINARY, "dim": 1},
        "partitive": {"type": BINARY, "dim": 1},
        "location": {"type": BINARY, "dim": 1},
        "instigation": {"type": BINARY, "dim": 1},
        "existed_after": {"type": BINARY, "dim": 1},
        "time": {"type": BINARY, "dim": 1},
        "awareness": {"type": BINARY, "dim": 1},
        "change_of_location": {"type": BINARY, "dim": 1},
        "manner": {"type": BINARY, "dim": 1},
        "sentient": {"type": BINARY, "dim": 1},
        "was_for_benefit": {"type": BINARY, "dim": 1},
        "change_of_state_continuous": {"type": BINARY, "dim": 1},
        "existed_during": {"type": BINARY, "dim": 1},
        "change_of_possession": {"type": BINARY, "dim": 1},
        "existed_before": {"type": BINARY, "dim": 1},
        "volition": {"type": BINARY, "dim": 1},
        "change_of_state": {"type": BINARY, "dim": 1},
    },
    "distributivity": {"distributive": {"type": BINARY, "dim": 1}},
}

DOCUMENT_EDGE_ANNOTATION_ATTRIBUTES = {
    "mereology": {"containment": {"type": BINARY, "dim": 1}},
    "time": {"temporal-relation": {"type": NOMINAL, "dim": 10}},
}
