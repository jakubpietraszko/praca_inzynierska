import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../../environments/environment';

export interface GradientStatisticsRequest {
  model: string;
  dataset: string;
  heatmap: string;
  percentage: number;
  label: string;
}

export interface GradientStatisticsResponse {
  success: boolean;
  list_of_rate_of_increase_in_scores: number[];
  list_of_average_drop: number[];
}

export interface ShapeMetricsLabelRequest {
  model: string;
  dataset: string;
  heatmap: string;
  label: string;
  percentage: number;
}

export interface ShapeMetricsLabelResponse {
  success: boolean;
  areas: number[];
  perimeters: number[];
  circularities: number[];
  rectangularities: number[];
}

@Injectable({
  providedIn: 'root',
})
export class HeatmapsStatisticsLabel {
  private http = inject(HttpClient);

  getScore(
    request: GradientStatisticsRequest,
  ): Observable<GradientStatisticsResponse> {
    return this.http.post<GradientStatisticsResponse>(
      `${environment.apiUrl}/heatmap/heatmaps-statistics-label-many`,
      request,
    );
  }

  getShapeScore(
    request: ShapeMetricsLabelRequest,
  ): Observable<ShapeMetricsLabelResponse> {
    return this.http.post<ShapeMetricsLabelResponse>(
      `${environment.apiUrl}/heatmap/heatmaps-statistics-label-shapes`,
      request,
    );
  }
}
