"""Functions and objects for replicating the comformity analysis."""

import enum
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy


class ConfidenceLevel(enum.Enum):
  """Valid confidence levels for filtering the human Likert responses."""
  ALL: str = 'All confidence ratings combined.'
  HIGH: str = 'High confidence responses.'
  LOW: str = 'Low confidence responses.'


def _get_image_agreement(
    prediction: int,
    labels: np.ndarray,
    confidence: ConfidenceLevel,
    ) -> Optional[float]:
  """For a single image and confidence level, compute the human-model agreement.

  The model produces a single prediction. The percentage of human labels which
  agree with the model for this image is computed. This is conditioned on
  the confidence level of the human labels, which can either be high, low or all
  confidence levels.

  Args:
    prediction: A binary model prediction (0 or 1).
    labels: An array of human Likert score labels between 1 and 5. Each of these
      is from a different human rater.
    confidence: Indicates whether to condition the agreement score on low or
      high human scores, or use all scores.

  Returns:
    A percentage human-model agreement score for the given image.
  """

  # Make the human rating filters conditioned on the desired confidence level.
  if confidence is ConfidenceLevel.HIGH:
    # Combine the confidence filters for both decision types.
    combined_filter = (labels == 1) + (labels == 5)
    # Apply combined filter to the labels.
    labels = labels[combined_filter]
  elif confidence is ConfidenceLevel.LOW:
    # Combine the confidence filters for both decision types.
    combined_filter = (labels == 2) + (labels == 4)
    # Apply combined filter to the labels.
    labels = labels[combined_filter]
  elif confidence is ConfidenceLevel.ALL:
    pass
  else:
    raise ValueError('Confidence must be a valid ConfidenceLevel.')

  # Check that there are some labels left.
  if list(labels):
    # Binarize the labels
    labels_bin = labels > 3
    # Compute the raw agreement count between human labels and model prediction.
    agreement_count = np.sum(labels_bin == prediction)
    # Convert to percentage agreement.
    agreement = agreement_count/len(labels)
    return np.round(agreement, decimals=4)
  else:
    return None


def get_delta_agreement(
    confidence: ConfidenceLevel,
    df: pd.DataFrame,
    ) -> Sequence[float]:
  """Returns the imagewise change in human-model agreement.

  This change in agreement is between the baseline no message (NM) condition
  and the prediction only (PO) condition. The aim here is to explore whether
  human raters are more likely to produce ratings that concur with the model
  when they are given the model predictions - i.e. the extent to which they
  use this information when making their own decisions.

  Args:
    confidence: Indicates whether to condition the agreement score on low or
      high human scores, or use all scores.
    df: A pandas dataframe containing the imagewise data.

  Returns:
    A set of 'delta agreement' scores, one per image. These are simply the
    human-model agreement scores when mode predictions are available, versus
    when theuy are not.
  """
  delta_agreement = []
  for _, row in df.iterrows():
    agreement_nm = []
    agreement_po = []
    for form in range(4):
      # Get the condition mapped to this data subdivision.
      form_con = row[f'condition_form_{form}']
      # Get the human labels for this data subdivision.
      labels = np.array(row[f'human_ratings_form_{form}'])
      # Get the model prediction.
      prediction = row['model_prediction']

      agreement = _get_image_agreement(prediction, labels, confidence)
      # If the condition is either the no message or prediction only, get
      # the human_model agreement, and append to the relevant list.
      if form_con == 'NM' and agreement is not None:
        agreement_nm.append(agreement)
      elif form_con == 'PO' and agreement is not None:
        agreement_po.append(agreement)
    # If agreement scores are found in both of these conditions, compute the
    # mean advantage for the PO condition, and append to the overall result
    # list.
    if agreement_nm and agreement_po:
      delta_agreement.append(np.mean(agreement_po) - np.mean(agreement_nm))
  return delta_agreement


def get_conformity_stats(df: pd.DataFrame) -> Tuple[float, float]:
  """Compute statistics for high vs. low confidence conformity comparison.

  Args:
    df: Dataframe containing the imagewise data.

  Returns:
    A tuple containing the t and p values from a t-test for
    independent samples.
  """
  high = get_delta_agreement(ConfidenceLevel.HIGH, df)
  low = get_delta_agreement(ConfidenceLevel.LOW, df)
  return scipy.stats.ttest_ind(low, high)
