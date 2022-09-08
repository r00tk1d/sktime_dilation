# -*- coding: utf-8 -*-
"""Dictionary based time series classifiers."""
__all__ = [
    "IndividualBOSS",
    "IndividualBOSSDilation",
    "BOSSEnsemble",
    "BOSSEnsembleDilation",
    "ContractableBOSS",
    "TemporalDictionaryEnsemble",
    "IndividualTDE",
    "WEASEL",
    "MUSE",
]

from sktime.classification.dictionary_based._boss import BOSSEnsemble, IndividualBOSS
from sktime.classification.dictionary_based._boss_dilation import BOSSEnsembleDilation, IndividualBOSSDilation
from sktime.classification.dictionary_based._cboss import ContractableBOSS
from sktime.classification.dictionary_based._muse import MUSE
from sktime.classification.dictionary_based._tde import (
    IndividualTDE,
    TemporalDictionaryEnsemble,
)
from sktime.classification.dictionary_based._weasel import WEASEL
