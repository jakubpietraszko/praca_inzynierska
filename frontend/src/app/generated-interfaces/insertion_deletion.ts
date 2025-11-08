/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface InsertionDeletionRequest {
  model: string;
  dataset: string;
  images: LabelFilenameTarget[];
  heatmap: string;
  threshold: number;
}
export interface LabelFilenameTarget {
  label: string;
  filename: string;
  target?: number | null;
}
export interface InsertionDeletionResponse {
  insertion_scores: ListOfScores[];
  deletion_scores: ListOfScores[];
  insertion_auc: number;
  deletion_auc: number;
}
export interface ListOfScores {
  scores: number[];
}
