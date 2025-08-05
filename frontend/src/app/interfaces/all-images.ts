export interface AllImagesRequest {
  dataset: string;
  label: string;
}

export interface AllImagesResponse {
    images: string[];
}