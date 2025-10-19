/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface HeatmapsStatisticsDatasetRequest {
  model: string;
  dataset: string;
}
export interface HeatmapsStatisticsDatasetResponse {
  average_drop: number[];
  rate_of_increase: number[];
  success: boolean;
}
