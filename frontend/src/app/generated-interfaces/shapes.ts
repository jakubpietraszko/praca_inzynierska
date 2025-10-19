/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface ShapeMetricsDatasetRequest {
  model: string;
  dataset: string;
  heatmap: string;
  percentage: number;
}
export interface ShapeMetricsDatasetResponse {
  success: boolean;
  areas: number[];
  perimeters: number[];
  circularities: number[];
  rectangularities: number[];
}
export interface ShapeMetricsLabelRequest {
  model: string;
  dataset: string;
  heatmap: string;
  percentage: number;
  label: string;
}
export interface ShapeMetricsLabelResponse {
  success: boolean;
  areas: number[];
  perimeters: number[];
  circularities: number[];
  rectangularities: number[];
}
