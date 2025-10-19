/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface HeatmapsStatisticsImageRequest {
  model: string;
  dataset: string;
  label: string;
  heatmap: string;
  image: string;
}
export interface HeatmapsStatisticsImageResponse {
  deletion: number;
  insertion: number;
  overall: number;
  success: boolean;
}
