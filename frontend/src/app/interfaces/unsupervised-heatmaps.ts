export interface UnsupervisedHeatmapsRequest {
  model: string;
  dataset: string;
  label: string;
  heatmap: string;
  type: string;
}

export interface FilenameCoords {
  filename: string;
  coordX: number;
  coordY: number;
}

export interface UnsupervisedHeatmapsResponse {
  filenames_coords: FilenameCoords[];
  success: boolean;
}
