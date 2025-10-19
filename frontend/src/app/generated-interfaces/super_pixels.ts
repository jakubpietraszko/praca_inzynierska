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
export interface SuperPixelsRequest {
  model: string;
  dataset: string;
  images: LabelFilename[];
  superPixels: number;
  overwrite: boolean;
}
export interface SuperPixelsResponse {
  success: boolean;
  superPixels: number[];
}
