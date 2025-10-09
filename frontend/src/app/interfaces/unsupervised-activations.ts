export interface UnsupervisedActivationsRequest {
  model: string;
  dataset: string;
  label: string;
  filename: string;
  type: string;
}

export interface LabelFilenamesCoordinates {
  label: string;
  filename: string;
  coordX: number;
  coordY: number;
}

export interface UnsupervisedActivationsResponse {
  dataset: string;
  label_filenames_coords: LabelFilenamesCoordinates[];
  success: boolean;
}

export interface GeneratingActivationsRequest {
  model: string;
  dataset: string;
  label: string;
  filename: string;
}

export interface GeneratingActivationsResponse {
  success: boolean;
  count: number;
}
