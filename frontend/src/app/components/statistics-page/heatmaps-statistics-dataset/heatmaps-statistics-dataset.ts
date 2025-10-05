import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';

export interface GradientStatisticsRequest {
  model: string;
  dataset: string;
  heatmap: string;
  percentage: number;
}

export interface GradientStatisticsResponse {
  success: boolean;
  list_of_rate_of_increase_in_scores: number[];
  list_of_average_drop: number[];
}

@Injectable({
  providedIn: 'root',
})
export class HeatmapsStatisticsDataset {
  private http = inject(HttpClient);

  getScore(
    request: GradientStatisticsRequest
  ): Observable<GradientStatisticsResponse> {
    return this.http.post<GradientStatisticsResponse>(
      `${environment.apiUrl}/heatmap/heatmaps-statistics-dataset-many`,
      request
    );
  }
}
