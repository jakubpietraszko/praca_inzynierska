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
}
