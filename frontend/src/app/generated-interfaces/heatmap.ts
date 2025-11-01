/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface HeatmapRequest {
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
export interface HeatmapResponse {
  success: boolean;
}
export interface HeatmapStatisticRequest {
  model: string;
  dataset: string;
  images: LabelFilenameTarget[];
  overwrite: boolean;
  heatmap: string;
  threshold: number;
  threshold_type: string;
}
export interface HeatmapStatisticResponse {
  success: boolean;
  softmax_scores: ListOfScores[];
}
export interface ListOfScores {
  scores: number[];
}
export interface HeatmapStatisticsDatasetManyRequest {
  model: string;
  dataset: string;
  heatmap: string;
  percentage: number;
}
export interface HeatmapStatisticsDatasetManyResponse {
  success: boolean;
  list_of_rate_of_increase_in_scores: number[];
  list_of_average_drop: number[];
}
export interface HeatmapStatisticsDatasetRequest {
  model: string;
  dataset: string;
  heatmap: string;
  threshold: number;
  percentage: number;
}
export interface HeatmapStatisticsDatasetResponse {
  success: boolean;
  rate_of_increase_in_scores: number;
  average_drop: number;
}
export interface HeatmapStatisticsLabelManyRequest {
  model: string;
  dataset: string;
  heatmap: string;
  percentage: number;
  label: string;
}
export interface HeatmapStatisticsLabelManyResponse {
  success: boolean;
  list_of_rate_of_increase_in_scores: number[];
  list_of_average_drop: number[];
}
export interface HeatmapStatisticsLabelRequest {
  model: string;
  dataset: string;
  heatmap: string;
  threshold: number;
  percentage: number;
  label: string;
}
export interface HeatmapStatisticsLabelResponse {
  success: boolean;
  rate_of_increase_in_scores: number;
  average_drop: number;
}
