/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface ModelDatasetI {
  model: string;
  dataset: string;
}
export interface ModelDatasetLabelHeatmapI {
  model: string;
  dataset: string;
  label: string;
  heatmap: string;
}
export interface NumeratorDenominator {
  numerator: {
    [k: string]: number;
  };
  denominator: number;
}
export interface Response {
  success: boolean;
}
