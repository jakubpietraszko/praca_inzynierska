/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface ActivationsRequest {
  model: string;
  dataset: string;
  label: string;
  filename: string;
}
export interface ActivationsResponse {
  success: boolean;
  count: number;
}
export interface AnyActivationsRequest {
  model: string;
  dataset: string;
  label: string;
  filename: string;
  ref: string;
}
