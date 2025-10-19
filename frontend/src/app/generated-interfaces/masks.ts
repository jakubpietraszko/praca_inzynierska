/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface LabelFilename {
  label: string;
  filename: string;
}
export interface MasksRequest {
  model: string;
  dataset: string;
  images: LabelFilename[];
  count: number;
  overwrite: boolean;
}
export interface MasksResponse {
  success: boolean;
}
