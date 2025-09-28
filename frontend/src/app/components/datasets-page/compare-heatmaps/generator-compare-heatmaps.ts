import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';

export interface LabelFilenameTarget {
  label: string;
  filename: string;
  target: string | null;
}

export interface CompareHeatmapsRequest {
  dataset: string;
  model: string;
  images: LabelFilenameTarget[];
  overwrite: boolean;
}

export interface CompareHeatmapsResponse {
  success: boolean;
}

@Injectable({
  providedIn: 'root',
})
export class GeneratorCompareHeatmaps {
  http = inject(HttpClient);

  compareHeatmaps(
    request: CompareHeatmapsRequest
  ): Observable<CompareHeatmapsResponse> {
    return this.http.post<CompareHeatmapsResponse>(
      `${environment.apiUrl}/heatmap/all-heatmaps`,
      request
    );
  }
}
