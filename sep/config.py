"""Project-wide constants: classes, label mappings, audio format.

Design decision (option b): the model is trained on 4 *query* classes only.
"other" is never predicted directly; it is computed at inference time as

    other = mix - (vocals + drums + guitar + piano)

so the 5 stems the user can pick always sum back to the original mix.
"""

SAMPLE_RATE = 44100
CHANNELS = 2

# Classes the conditioned model can be queried for (index = embedding id).
QUERY_CLASSES = ["vocals", "drums", "guitar", "piano"]
QUERY_INDEX = {c: i for i, c in enumerate(QUERY_CLASSES)}

# Classes shown to the user. "other" = residual.
USER_CLASSES = QUERY_CLASSES + ["other"]

# Labels a stem can carry inside the manifest.
#   one of USER_CLASSES  -> a clean stem of that class
#   "ambiguous"          -> a stem that MAY contain some query classes
#                           (listed in the stem's `may_contain`), e.g. MUSDB "other"
STEM_LABELS = USER_CLASSES + ["ambiguous"]

# ---------------------------------------------------------------- MUSDB18-HQ
# Folder per song: mixture.wav vocals.wav drums.wav bass.wav other.wav
# MUSDB "other" contains guitar / piano / synths mixed together, so it cannot be a
# clean "other" stem and it blocks guitar/piano queries on any mix containing it.
MUSDB_MAP = {
    "vocals": ("vocals", []),
    "drums": ("drums", []),
    "bass": ("other", []),
    "other": ("ambiguous", ["guitar", "piano"]),
}

# ------------------------------------------------------------------ MoisesDB
# Folder per track (uuid) with one sub-folder per top-level stem, each holding
# one or more wav files (summed). Unknown folder names fall back to "other".
MOISESDB_MAP = {
    "vocals": "vocals",
    "drums": "drums",
    "percussion": "drums",      # convention: drums class includes aux percussion
    "guitar": "guitar",
    "piano": "piano",
    "bass": "other",
    "other_keys": "other",      # organ / synth keys -> other (keeps "piano" tight)
    "bowed_strings": "other",
    "wind": "other",
    "other_plucked": "other",
    "other": "other",
}

# ------------------------------------------------------------------ Slakh2100
# TrackXXXXX/metadata.yaml lists stems with `inst_class`; audio in stems/SXX.flac.
# Slakh has no vocals.
SLAKH_MAP = {
    "Drums": "drums",
    "Guitar": "guitar",
    "Piano": "piano",
}
SLAKH_DEFAULT = "other"


# ------------------------------------------------------- hard negatives (run4)
# "other" sub-instruments that sound like the query class: sustained / harmonic parts the
# listening test showed leaking into piano and guitar (strings, pads, organ, synth, winds).
SUSTAINED_OTHER = {
    # MoisesDB folders
    "bowed_strings", "other_keys", "wind", "other_plucked",
    # Slakh inst_class
    "Strings", "Strings (continued)", "Ensemble", "Organ", "Synth Pad", "Synth Lead",
    "Brass", "Reed", "Pipe", "Chromatic Percussion",
}
# for these queries, a "hard" mix also contains the paired class
CONFUSABLE_PAIR = {"piano": "guitar", "guitar": "piano"}
