/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface Color {
  r: number;
  g: number;
  b: number;
}
export interface Coords {
  x: number;
  y: number;
}
export interface UnsupervisedLabelItem {
  filename: string;
  coords: Coords;
  color: Color;
}
export interface UnsupervisedLabelRequest {
  dataset: string;
  label: string;
}
export interface UnsupervisedLabelResponse {
  items: UnsupervisedLabelItem[];
  success: boolean;
}
