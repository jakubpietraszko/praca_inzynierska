import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';

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
  success: boolean;
  filenames_coords: FilenameCoords[];
}

@Injectable({
  providedIn: 'root',
})
export class GeneratorUnsupervisedHeatmaps {
  http = inject(HttpClient);

  unsupervisedHeatmaps(
    request: UnsupervisedHeatmapsRequest,
  ): Observable<UnsupervisedHeatmapsResponse> {
    return this.http.post<UnsupervisedHeatmapsResponse>(
      `${environment.apiUrl}/heatmap/unsupervised-heatmaps`,
      request,
    );
  }
}
