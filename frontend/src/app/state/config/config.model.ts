export interface XAIMethodConfig {
  epsilon?: number;
  aggregation?: string;
  group_size?: number;
  k_size?: number;
  sigma?: number;
  n_samples?: number;
  noise_level?: number;
  baseline?: number;
  steps?: number;
  super_pixels?: number;
  segmentation?: string;
  gamma?: number;
  alpha?: number;
  beta?: number;
}

export interface ConfigModel {
  methods: { [methodName: string]: XAIMethodConfig };
}

export interface ConfigState {
  config: ConfigModel | null;
  loading: boolean;
  error: string | null;
}
