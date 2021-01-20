## Event Structure Annotations

This directory contains UDS-EventStructure annotations used to train the model, which are not included in UDS v0.1.0, though they will be included in the forthcoming v0.2.0. All other UDS annotations are loaded automatically with the corpus.

In collecting annotations for situation and subevent duration for UDS-EventStructure, we used a different coding scheme for the durration values ("seconds," "minutes," etc.) than was used for the UDS-Time annotations. Specifically, the former go from longest duration (0 - "effectively forever") to shortest duration (11 - "effectively no time at all"), whereas the latter go in the reverse order (and have fewer categories). The annotations in `natural_parts_and_telicity.json` preserve the coding used for data collection, while `natural_parts_and_telicity_corrected.json` reverses it, which makes for easier comparison to the UDS-Time duration annotations during analysis.
