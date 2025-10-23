import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';

export interface UnsupervisedActivationsRequest {
  model: string;
  dataset: string;
  label: string;
  filename: string;
  type: string;
}

export interface UnsupervisedAnyActivationsRequest
  extends UnsupervisedActivationsRequest {
  ref: string;
}

export interface ImageCoordinate {
  label: string;
  filename: string;
  coordX: number;
  coordY: number;
}

export interface UnsupervisedActivationsResponse {
  dataset: string;
  label_filenames_coords: ImageCoordinate[];
  success: boolean;
}

export interface UnsupervisedAnyActivationsResponse
  extends UnsupervisedActivationsResponse {}

@Injectable({
  providedIn: 'root',
})
export class GeneratorUnsupervisedActivations {
  http = inject(HttpClient);

  unsupervisedActivations(
    request: UnsupervisedActivationsRequest,
  ): Observable<UnsupervisedActivationsResponse> {
    return this.http.post<UnsupervisedActivationsResponse>(
      `${environment.apiUrl}/heatmap/unsupervised-activations`,
      request,
    );
  }

  unsupervisedAnyActivations(
    request: UnsupervisedAnyActivationsRequest,
  ): Observable<UnsupervisedAnyActivationsResponse> {
    return this.http.post<UnsupervisedAnyActivationsResponse>(
      `${environment.apiUrl}/heatmap/unsupervised-any-activations`,
      request,
    );
  }
}
