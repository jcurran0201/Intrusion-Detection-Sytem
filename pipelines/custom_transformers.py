"""
pipeline/custom_transformers.py

Custom sklearn/imblearn transformers used inside the saved model pipeline.

CappedSMOTE was originally defined directly in the training notebook, which
means it was pickled with __module__ == '__main__'. That works fine inside
the notebook, but breaks the moment the model is unpickled in a different
process (e.g. uvicorn's --reload subprocess, or any other entry point) —
Python has no idea what 'CappedSMOTE' means there.

Giving it a real, importable home here — and registering it into __main__
in predict.py before joblib.load() — fixes that permanently regardless of
which script/process ends up loading the model.

Note: imblearn Pipelines only run the resampling step during .fit(), not
during .predict()/.predict_proba() — so this class is inert at inference
time. It only needs to exist so pickle can resolve the class reference.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.base import BaseSampler


class CappedSMOTE(BaseSampler):
    """
    SMOTE wrapper that ensures no minority class gets oversampled past
    `cap_ratio` (default 10%) of the majority class's sample count —
    prevents near-duplicate synthetic samples on very small classes.
    """
    _sampling_type = 'over-sampling'  # we make it clear that we are oversampling to add minority classes
    _parameter_constraints = {}

    def __init__(self, cap_ratio=0.10, random_state=42):
        super().__init__()
        self.cap_ratio = cap_ratio          # set the cap_ratio
        self.random_state = random_state    # set the random state

    def _fit_resample(self, X, y):
        counts = pd.Series(y).value_counts()          # how many values are in each classification
        majority_count = counts.max()                  # count of the majority class
        smote_target = int(majority_count * self.cap_ratio)  # count of majority class * 0.1
        strategy = {                                    # sampling_strategy for the SMOTE
            label: max(count, smote_target)              # sets target to whichever is larger, current count/smote target.
                                                          # prevents accidentally telling SMOTE to undersample a class that's already above the target
            for label, count in counts.items()           # loops through each attack class and its current sample count
            if count < smote_target                      # only includes classes that are below the target, skipping classes that already have enough samples
        }
        if not strategy:            # we return X,y if we dont need to add to a specific class(es)
            return X, y
        sm = SMOTE(sampling_strategy=strategy, random_state=self.random_state)  # set SMOTE as a variable
        return sm.fit_resample(X, y)  # fit and resample


SMOTE_STEP = ('smote', CappedSMOTE(cap_ratio=0.10, random_state=42))  # completed class in SMOTE
