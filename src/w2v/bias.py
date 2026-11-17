"""Gender-direction probing (Bolukbasi et al. 2016, "Man is to Computer Programmer as Woman is to Homemaker?").

The gender direction :math:`g` is the first principal component of the
*centred* definitional pairs: for each pair (f, m) we take
:math:`\\hat f - \\mu` and :math:`\\hat m - \\mu` with :math:`\\mu = (\\hat f + \\hat m)/2`
(unit vectors), stack them, and run PCA. Direct bias of a set of gender-neutral
words :math:`N` is :math:`\\frac{1}{|N|}\\sum_{w\\in N} |\\cos(w, g)|^c`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .embeddings import Embeddings

DEFINITIONAL_PAIRS: list[tuple[str, str]] = [
    ("she", "he"),
    ("her", "his"),
    ("woman", "man"),
    ("mary", "john"),
    ("herself", "himself"),
    ("daughter", "son"),
    ("mother", "father"),
    ("gal", "guy"),
    ("girl", "boy"),
    ("female", "male"),
]

# Gender-neutral occupation words (subset of the Bolukbasi et al. profession list).
OCCUPATIONS: list[str] = [
    "nurse", "homemaker", "receptionist", "librarian", "socialite", "hairdresser", "nanny", "stylist",
    "housekeeper", "designer", "dancer", "secretary", "teacher", "singer", "model", "therapist",
    "artist", "writer", "novelist", "poet", "journalist", "editor", "photographer", "actor",
    "doctor", "physician", "surgeon", "lawyer", "judge", "professor", "scientist", "physicist",
    "chemist", "mathematician", "philosopher", "economist", "engineer", "architect", "programmer",
    "banker", "accountant", "manager", "businessman", "entrepreneur", "politician", "diplomat",
    "soldier", "officer", "captain", "commander", "pilot", "sailor", "farmer", "carpenter",
    "mechanic", "plumber", "builder", "electrician", "firefighter", "president", "boss", "warrior",
    "magician", "athlete", "coach", "footballer", "student", "clerk", "cook", "chef", "baker",
]


def gender_direction(emb: Embeddings, pairs: Sequence[tuple[str, str]] = DEFINITIONAL_PAIRS) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit gender direction (oriented so that ``she`` projects positively) and the PCA spectrum.

    The spectrum is the explained-variance ratio of the top components; a dominant
    first component means gender is (approximately) one-dimensional in the space.
    """
    rows = []
    for f, m in pairs:
        if f in emb and m in emb:
            vf, vm = emb.unit[emb.index[f]], emb.unit[emb.index[m]]
            mu = (vf + vm) / 2
            rows += [vf - mu, vm - mu]
    X = np.stack(rows)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    g = vt[0]
    f0, m0 = next((f, m) for f, m in pairs if f in emb and m in emb)
    if (emb.unit[emb.index[f0]] - emb.unit[emb.index[m0]]) @ g < 0:
        g = -g
    var = s**2
    return g / np.linalg.norm(g), var / var.sum()


def projections(emb: Embeddings, words: Sequence[str], g: np.ndarray) -> dict[str, float]:
    """Cosine of each in-vocabulary word with the gender direction (positive = closer to *she*)."""
    return {w: float(emb.unit[emb.index[w]] @ g) for w in words if w in emb}


def direct_bias(emb: Embeddings, words: Sequence[str], g: np.ndarray, c: float = 1.0) -> float:
    """DirectBias_c of Bolukbasi et al.: mean |cos(w, g)|^c over neutral words."""
    p = np.array(list(projections(emb, words, g).values()))
    return float(np.mean(np.abs(p) ** c))


def neutralize(emb: Embeddings, words: Sequence[str], g: np.ndarray) -> Embeddings:
    """Hard-debias "neutralize" step: remove the gender component from ``words`` and renormalise."""
    vecs = emb.unit.copy()
    for w in words:
        if w in emb:
            i = emb.index[w]
            v = vecs[i] - (vecs[i] @ g) * g
            vecs[i] = v / np.linalg.norm(v)
    return Embeddings(emb.words, vecs)
