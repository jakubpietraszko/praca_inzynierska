/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface FetchActivationImageRequest {
  dataset: string;
  label: string;
  filename: string;
  model: string;
  image: string;
}
export interface FetchAnyActivationImageRequest {
  dataset: string;
  label: string;
  filename: string;
  model: string;
  ref: string;
  image: string;
}
export interface FetchImageRequest {
  dataset: string;
  label: string;
  filename: string;
}
export interface FetchImageResponse {
  image: string;
}
export interface FetchModelsDatasetsRequest {
  dataset: string;
  label: string;
  filename: string;
  model: string;
  type: string;
}
