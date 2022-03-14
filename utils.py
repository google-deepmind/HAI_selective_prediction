"""Utility analysis and plotting functions for the deferral project."""

import copy
from typing import List, Sequence, Tuple, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


# Model score above which the model classifies there being an animal present.
MODEL_DECISION_THRESHOLD = 0.1931999921798706

# Plotting settings.
COLOURS = ['#687566', '#4464AD', '#F5843D']
_BAR_KWARGS = {'width': 0.7, 'zorder': 1, 'edgecolor': 'k', 'linewidth': 1}
_ERRORBAR_KWARGS = {'zorder': 2, 'fmt': 'None', 'linewidth': 2, 'ecolor': 'k'}

CORRECTNESS_LABELS = ['1', '0']  # 1 is Correct, 0 is incorrect.


def correctness_to_accuracy(correctness: Sequence[int]) -> float:
  """Converts binary correctness scores to percentage accuracy."""
  return sum(correctness)*100 / float(len(correctness))


def get_95confidence(data: np.ndarray) -> None:
  """Returns the error magnitude for 95% confidence interval."""
  return (data.std() / np.sqrt(data.shape[0])) * 1.96


def likert_to_confidence(likerts: Sequence[int]) -> np.ndarray:
  """Converts likert values to a confidence score."""
  confidence = []
  mapping = {1: 2, 2: 1, 3: 0, 4: 1, 5: 2}  # Maps likert to a confidence value.
  for likert in likerts:
    confidence.append(mapping[likert])
  return np.mean(confidence)


def likert_to_correctness(
    likerts: Sequence[int],
    ground_truth: np.ndarray,
) -> List[int]:
  """Converts likert values to binary correctness scores.

  'Correct' means the human classified the image correctly, regardless of
  confidence in that classification. An 'unsure' Likert value of 3, is mapped
  to `incorrect`.

  Args:
    likerts: An array of integer Likert values in the set {1,2,3,4,5}, which
      correspond to confidence judgements in a classification decision. A value
      of 5 means high confidence that an animal was in the image and a value of
      1 means high confidence that an animal was not in the image.
    ground_truth: An array of binary labels indicating whether an animal was
      present in an image (=1) or not (=0).
  Returns:
    A list of binary correctness labels.
  """
  incorrect_label, correct_label = (0, 1)
  threshold_likert = 3
  correct = []
  for likert in likerts:
    if likert == threshold_likert:  # likert==3 is always incorrect
      correct.append(incorrect_label)
    elif ground_truth:
      # There is an animal in the image.
      if likert > threshold_likert:
        correct.append(correct_label)  # True positive.
      else:
        correct.append(incorrect_label)  # False negative.
    else:
      # There is not an image in the image.
      if likert > threshold_likert:
        correct.append(incorrect_label)  # True negative.
      else:
        correct.append(correct_label)  # False positive.
  return correct


def model_score_to_confidence(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
  """Converts model scores to a confidence metric."""
  return np.abs(scores - threshold)


def compute_imagewise_accuracy(
    data: pd.DataFrame,
    conditions: Sequence[str],
) -> Mapping[str, np.ndarray]:
  """Computes human accuracy on each image, when the model is correct/incorrect.

  Restricts data to specific testing conditions.
  Args:
    data: Imagewise dataframe.
    conditions: Only data from images presented in these named conditions will
      be included in the analysis.
  Returns:
    Human accuracy on each image for images examined under the specified
    conditions. Output as a dictionary which divides accuracy into images for
    which the model was correct or not.
  """
  accuracy = {'model_correct': [], 'model_incorrect': []}
  for _, row in data.iterrows():
    images_found = False
    human_labels = []
    if row['condition_form_0'] in conditions:
      human_labels.append(row['human_ratings_form_0'])
      images_found = True
    if row['condition_form_1'] in conditions:
      human_labels.append(row['human_ratings_form_1'])
      images_found = True
    if row['condition_form_2'] in conditions:
      human_labels.append(row['human_ratings_form_2'])
      images_found = True
    if row['condition_form_3'] in conditions:
      human_labels.append(row['human_ratings_form_3'])
      images_found = True

    if images_found:
      # Human accuracy.
      flat_labels = []
      for sublist in human_labels:
        for element in sublist:
          flat_labels.append(element)
      human_correct = likert_to_correctness(flat_labels, row['label'])
      human_accuracy = correctness_to_accuracy(human_correct)

      # Model performance.
      model_answer = (1 if row['model_score'] > MODEL_DECISION_THRESHOLD else 0)
      model_correct = (1 if model_answer == row['label'] else 0)

      if model_correct:
        accuracy['model_correct'].append(human_accuracy)
      else:
        accuracy['model_incorrect'].append(human_accuracy)

  return {key: np.asarray(value) for key, value in accuracy.items()}


def compute_human_and_model_scores(
    data: pd.DataFrame,
    conditions: Sequence[str],
) -> Tuple[List[float], List[float]]:
  """Computes human likert scores and model scores for each image.

  Restricts data to specific testing conditions.
  Args:
    data: Imagewise dataframe.
    conditions: Only data from images presented in these named conditions will
      be included in the analysis.

  Returns:
    Mean human likert scores and model scores on each image for the given
    conditions.
  """
  mean_likerts = []
  model_scores = []
  for _, row in data.iterrows():
    image_found = False
    human_labels = []
    if row['condition_form_0'] in conditions:
      human_labels.append(row['human_ratings_form_0'])
      image_found = True
    if row['condition_form_1'] in conditions:
      human_labels.append(row['human_ratings_form_1'])
      image_found = True
    if row['condition_form_2'] in conditions:
      human_labels.append(row['human_ratings_form_2'])
      image_found = True
    if row['condition_form_3'] in conditions:
      human_labels.append(row['human_ratings_form_3'])
      image_found = True

    if image_found:
      flat_labels = []
      for sublist in human_labels:
        for element in sublist:
          flat_labels.append(element)
      mean_likerts.append(np.mean(flat_labels))
      model_scores.append(row['model_score'])

  return mean_likerts, model_scores


def figure_5(
    *,
    means: Sequence[np.ndarray],
    errors: Sequence[np.ndarray],
    labels: Sequence['str'],
    ax: plt.Axes,
    colours: Tuple[str],
    ylim: Tuple[float, float] = (45.0, 70.0),
) -> None:
  """Plots figure 5: Human accuracy in each of the four conditions.

  Args:
    means: Mean human accuracy for each of the four conditions.
    errors: Error magnitudes for each of our four conditions.
    labels: Labels for the x-axis.
    ax: Axes on which to plot.
    colours: Three colours for plotting with.
    ylim: y-axis limits.
  """
  xlabel_locations = np.arange(len(labels))
  handles = []
  for xlabel_location, mean, error in zip(xlabel_locations, means, errors):
    h = ax.bar(xlabel_location, 100.*mean, color=colours[0], **_BAR_KWARGS)
    handles.append(h)
    ax.errorbar(xlabel_location, 100.*mean, yerr=100.*error, **_ERRORBAR_KWARGS)
  ax.set_ylim(ylim)
  ax.set_xlim([-0.6, 3.6])
  ax.set_xticks(xlabel_locations)
  ax.set_xticklabels(labels)
  ax.hlines(50, -1, 5, color='darkgrey', linestyles='dashed', zorder=6)
  ax.set_ylabel('Human accuracy (%)')
  ax.set_xlabel('Messaging condition')


def figure_6(
    *,
    all_conformity: np.ndarray,
    sure_conformity: np.ndarray,
    unsure_conformity: np.ndarray,
    colours: Sequence[str],
    ) -> None:
  """Plots figure 6: Conformity analysis.

  We measure conformity as the difference in human-model agreement when the
  humans have no access to the model prediction, vs. when they do. The relative
  increase in agreement in the latter condition indicates the extent to which
  people integrate this information into their own judgments. Here we only
  investigate the conditions with no deferral message (NM and PO), to avoid any
  additional complicating factors of this message.

  As previous research indicates that people are more likely to conform when
  they are less confident in their own judgments, we additionally investigate
  conformity when the human judgments are low vs. high confidence.

  We report a statistically significant difference consistent with prior
  research, showing that conformity is higher when human confidence is low.

  Args:
    all_conformity: Array of all conformity scores regardless of confidence.
    sure_conformity: High confidence conformity scores.
    unsure_conformity: Low confidence conformity scores.
    colours: Three colours for plotting with.
  """

  # Run t test with independent samples to compare high and low confidence
  # conformity.
  t, p = scipy.stats.ttest_ind(unsure_conformity, sure_conformity)
  print('Difference between low and high confidence conformity: '
        f't={t:.2f}, p={p:.3f}')
  fig = plt.figure(figsize=(3.5, 4))
  ax = plt.gca()
  xpos = [1, 2, 3]
  plt.xticks(xpos)
  ax.set_xticklabels(['all', 'high', 'low'])
  conditions = [all_conformity, sure_conformity, unsure_conformity]
  plt.xlabel('Confidence')
  plt.ylabel('Conformity')
  violins = ax.violinplot(
      conditions,
      widths=0.4,
      showextrema=False,
  )
  # Add specified colours to the violin plots.
  for index, pc in enumerate(violins['bodies']):
    pc.set_facecolor(colours[index])
    pc.set_alpha(0.6)

  # Add mean and standard error to each plot
  for index, condition in enumerate(conditions):
    plt.scatter(
        index+1,
        np.mean(condition),
        color='k',
        )
    plt.errorbar(
        index + 1,
        np.mean(condition),
        yerr=np.std(condition) / np.sqrt(len(condition)) * 1.96,
        color='k',
        linewidth=2,
    )

  fig.patch.set_facecolor('white')


def figure_7a(
    *,
    means: Sequence[Sequence[np.ndarray]],
    errors: Sequence[Sequence[np.ndarray]],
    labels: Sequence[str],
    ax: plt.Axes,
    colours: Sequence[str],
    ylim: Tuple[float, float] = (35.0, 80.0),
    xlim: Tuple[float, float] = (-0.6, 3.6),
) -> None:
  """Plots figure 7A:  Human accuracy, subdivided based on model correctness.

  Args:
    means: Mean human accuracy for each condition, when model was also correct
      vs incorrect.
    errors: Error magnitudes for each of the conditions.
    labels: Labels for the x-axis.
    ax: Axes on which to plot.
    colours: Three colours for plotting with.
    ylim: y-axis limits.
    xlim: x-axis limits.
  """
  correct_means, incorrect_means = means
  correct_errs, incorrect_errs = errors
  xlabel_locations = np.arange(len(labels))  # the xlabel locations
  bar_kwargs = copy.copy(_BAR_KWARGS)
  bar_kwargs['width'] = 0.35

  # Plot a line at chance performance.
  ax.hlines(50, -1, 5, color='darkgrey', linestyles='dashed', zorder=6)

  # Plot human accuracy when the model was correct.
  for xlabel_location, correct_mean, correct_err in zip(
      xlabel_locations,
      correct_means,
      correct_errs,
  ):
    h_correct = ax.bar(
        xlabel_location - bar_kwargs['width'] / 2,
        100. * correct_mean,
        label='Model correct',
        color=colours[0],
        **bar_kwargs,
    )
    ax.errorbar(
        xlabel_location - bar_kwargs['width'] / 2,
        100. * correct_mean,
        yerr=100. * correct_err,
        **_ERRORBAR_KWARGS,
    )

  # Plot human accuracy when the model was incorrect.
  for xlabel_location, incorrect_mean, incorrect_err in zip(
      xlabel_locations,
      incorrect_means,
      incorrect_errs,
  ):
    h_incorrect = ax.bar(
        xlabel_location + bar_kwargs['width'] / 2,
        100. * incorrect_mean,
        label='Model incorrect',
        color=colours[1],
        **bar_kwargs,
    )
    ax.errorbar(
        xlabel_location + bar_kwargs['width'] / 2,
        100. * incorrect_mean,
        yerr=100. * incorrect_err,
        **_ERRORBAR_KWARGS,
    )
  handles = [h_correct, h_incorrect]

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Human accuracy (%)')
  ax.set_xticks(xlabel_locations)
  ax.set_xticklabels(labels)
  ax.set_xlabel('Messaging condition')
  ax.legend(handles, ['Model correct', 'Model incorrect'])
  ax.set_ylim(ylim)
  ax.set_xlim(xlim)


def figure_7b(
    accuracy_correct: np.ndarray,
    accuracy_incorrect: np.ndarray,
    *,
    ax: plt.Axes,
    colours: Sequence[str],
    ylims: Tuple[float, float] = (0.0, 100.0),
    xlims: Tuple[float, float] = (0.5, 2.5),
) -> None:
  """Plots figure 7B: Human accuracy  in 'NM' and 'DO' conditions.

  'NM' = No message.
  'DO' = Deferred only.
  These conditions do not show the model predictions, yet we still see that
  humans have higher accuracy when the model is also correct. This is suggestive
  that humans and the model may find the same images difficult to classify.

  Args:
    accuracy_correct: Human accuracy on images for which the model was correct.
    accuracy_incorrect: Human accuracy on images for which model was incorrect.
    ax: Axes on which to plot.
    colours: Three colours for plotting with.
    ylims: y-axis limits.
    xlims: x-axis limits.
  """
  t, p = scipy.stats.ttest_ind(accuracy_correct, accuracy_incorrect)
  print(f'Difference : t={t:.2f}, p={p:.3f}')
  num_images = accuracy_correct.size
  means = [np.mean(accuracy_correct), np.mean(accuracy_incorrect)]
  std_devs = [np.std(accuracy_correct), np.std(accuracy_incorrect)]
  errors = std_devs / np.sqrt(num_images)  # std error of mean across images.

  x_ticks = [1, 2]
  violins = ax.violinplot(
      [accuracy_correct, accuracy_incorrect],
      widths=0.3,
      showextrema=False,
  )
  for index, pc in enumerate(violins['bodies']):
    pc.set_facecolor(colours[index+1])
    pc.set_alpha(0.6)
  ax.scatter(x_ticks, means, color='k')
  ax.errorbar(x_ticks, means, errors, **_ERRORBAR_KWARGS)

  # Add a dashed line to indicate chance performance.
  ax.hlines(50, -5, 5, color='darkgrey', linestyles='dashed', zorder=6)
  ax.set_ylim(ylims)
  ax.set_xlim(xlims)
  ax.set_xticks(x_ticks)
  ax.set_xticklabels(['Model\ncorrect', 'Model\nincorrect'])
  ax.set_ylabel('Human accuracy (%) \n (no message & defer only conditions)')
