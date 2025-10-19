/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface LabelFilenameCoords {
  label: string;
  filename: string;
  coordX: number;
  coordY: number;
}
export interface UnsupervisedActivationsRequest {
  model: string;
  dataset: string;
  label: string;
  filename: string;
  type: string;
}
export interface UnsupervisedActivationsResponse {
  dataset: string;
  label_filenames_coords: LabelFilenameCoords[];
  success: boolean;
}
