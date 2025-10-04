import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { environment } from '../../../../environments/environment';

export interface LabelFilenameTarget {
  label: string;
  filename: string;
  target: number | null;
}

export interface GradientStatisticsRequest {
  model: string;
  dataset: string;
  images: LabelFilenameTarget[];
  overwrite: boolean;
  heatmap: string;
  threshold: number;
  threshold_type: string;
}

export interface ListOfScore {
  scores: number[];
}

export interface GradientStatisticsResponse {
  softmax_scores: ListOfScore[];
}

@Injectable({
  providedIn: 'root',
})
export class HeatmapsStatisticsImage {
  private http = inject(HttpClient);

  getScore(
    request: GradientStatisticsRequest
  ): Observable<GradientStatisticsResponse> {
    return this.http.post<GradientStatisticsResponse>(
      `${environment.apiUrl}/heatmap/heatmap-statistics`,
      request
    );
  }
}
