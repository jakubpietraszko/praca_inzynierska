/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface ConfusionMatrix {
  [k: string]: ConfusionMatrixRow;
}
export interface ConfusionMatrixRow {
  [k: string]: number;
}
export interface ModelDatasetPredictionsRequest {
  model: string;
  dataset: string;
}
export interface ModelDatasetPredictionsResponse {
  predictions_proba: number[][];
  predictions: number[];
  true_labels: number[];
  labels: string[];
  confusion_matrix: ConfusionMatrix;
}
export interface ModelDatasetStatisticsRequest {
  model: string;
  dataset: string;
}
export interface ModelDatasetStatisticsResponse {
  accuracy: number;
  balanced_accuracy: number;
  confusion_matrix: ConfusionMatrix;
  roc_auc_per_class: {
    [k: string]: number;
  };
  roc_auc_macro: number;
  roc_auc_micro: number;
  roc_auc_weighted: number;
  support_per_class: {
    [k: string]: number;
  };
  precision_per_class: {
    [k: string]: number;
  };
  precision_macro: number;
  precision_micro: number;
  precision_weighted: number;
  recall_per_class: {
    [k: string]: number;
  };
  recall_macro: number;
  recall_micro: number;
  recall_weighted: number;
  f1_score_per_class: {
    [k: string]: number;
  };
  f1_score_macro: number;
  f1_score_micro: number;
  f1_score_weighted: number;
  jaccard_per_class: {
    [k: string]: number;
  };
  jaccard_macro: number;
  jaccard_micro: number;
  jaccard_weighted: number;
}
