export interface ModelLayersRequest {
  model: string;
}

export interface ModelLayersResponse {
  layers: string[];
}

export interface LayerConvRef {
  name: string;
  conv: boolean;
  ref: string;
}

export interface AllLayersResponse {
  layers: LayerConvRef[];
}
