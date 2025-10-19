/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface GradientRequest {
  model: string;
  dataset: string;
  images: LabelFilenameTarget[];
  overwrite: boolean;
}
export interface LabelFilenameTarget {
  label: string;
  filename: string;
  target?: number | null;
}
export interface GradientResponse {
  success: boolean;
}
export interface GradientStatisticRequest {
  model: string;
  dataset: string;
  images: LabelFilenameTarget[];
  overwrite: boolean;
  heatmap: string;
  threshold: number;
  threshold_type: string;
}
export interface GradientStatisticResponse {
  success: boolean;
  softmax_scores: ListOfScores[];
}
export interface ListOfScores {
  scores: number[];
}
export interface GradientStatisticsDatasetManyRequest {
  model: string;
  dataset: string;
  heatmap: string;
  percentage: number;
}
export interface GradientStatisticsDatasetManyResponse {
  success: boolean;
  list_of_rate_of_increase_in_scores: number[];
  list_of_average_drop: number[];
}
export interface GradientStatisticsDatasetRequest {
  model: string;
  dataset: string;
  heatmap: string;
  threshold: number;
  percentage: number;
}
export interface GradientStatisticsDatasetResponse {
  success: boolean;
  rate_of_increase_in_scores: number;
  average_drop: number;
}
export interface GradientStatisticsLabelManyRequest {
  model: string;
  dataset: string;
  heatmap: string;
  percentage: number;
  label: string;
}
export interface GradientStatisticsLabelManyResponse {
  success: boolean;
  list_of_rate_of_increase_in_scores: number[];
  list_of_average_drop: number[];
}
export interface GradientStatisticsLabelRequest {
  model: string;
  dataset: string;
  heatmap: string;
  threshold: number;
  percentage: number;
  label: string;
}
