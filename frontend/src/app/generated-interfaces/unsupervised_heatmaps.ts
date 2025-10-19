/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface FilenameCoords {
  filename: string;
  coordX: number;
  coordY: number;
}
export interface UnsupervisedHeatmapsRequest {
  model: string;
  dataset: string;
  label: string;
  heatmap: string;
  type: string;
}
export interface UnsupervisedHeatmapsResponse {
  filenames_coords: FilenameCoords[];
  success: boolean;
}
