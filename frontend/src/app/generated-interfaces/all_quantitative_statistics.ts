/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface AllQuantitativeStatisticsRequest {
  model: string;
  dataset: string;
}
export interface AllQuantitativeStatisticsResponse {
  success: boolean;
  confusion_dict: {
    [k: string]: {
      [k: string]: number;
    };
  };
}
