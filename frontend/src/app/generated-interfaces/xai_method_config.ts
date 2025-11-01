/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface ConfigModel {
  methods?: {
    [k: string]: XAIMethodConfig;
  };
}
export interface XAIMethodConfig {
  epsilon?: number | null;
  aggregation?: string | null;
  group_size?: number | null;
  k_size?: number | null;
  sigma?: number | null;
  n_samples?: number | null;
  noise_level?: number | null;
  baseline?: number | null;
  steps?: number | null;
  super_pixels?: number | null;
  segmentation?: string | null;
  gamma?: number | null;
  alpha?: number | null;
  beta?: number | null;
}
