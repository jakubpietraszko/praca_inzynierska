export interface FetchImageRequest {
    dataset: string;
    label: string;
    filename: string;
}

export interface FetchImageResponse {
    image: string;
}