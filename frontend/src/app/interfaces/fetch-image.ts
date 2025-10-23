export interface FetchImageRequest {
  dataset: string;
  label: string;
  filename: string;
}

export interface FetchImageResponse {
  image: string;
}

export interface FetchActivationRequest extends FetchImageRequest {
  model: string;
  image: string;
}

export interface FetchAnyActivationRequest extends FetchActivationRequest{
  ref: string;
}

export interface FetchModelsDatasetsRequest extends FetchImageRequest {
  model: string;
  type: string;
}
