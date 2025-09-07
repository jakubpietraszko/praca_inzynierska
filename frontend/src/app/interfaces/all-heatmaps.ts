export interface AllHeatmapsResponse {
    heatmaps: string[]
}

export interface LabelFilenameTarget {
    label: string;
    filename: string;
    target: string | null;
}

export interface HeatmapRequest {
    dataset: string;
    model: string;
    images: LabelFilenameTarget[];
    overwrite: boolean;
}

export interface HeatmapResponse {
    success: boolean;
}
