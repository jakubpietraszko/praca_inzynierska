import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';

export interface QuantitativeStatisticsRequest {
  dataset: string;
  model: string;
}

export interface QuantitativeStatisticsResponse {
  success: boolean;
  confusion_dict: Record<string, Record<string, number>>;
}

export interface HeatmapsStatisticsRequest {
  dataset: string;
  model: string;
  heatmap: string;
}

@Injectable({
  providedIn: 'root',
})
export class GeneratorStatistics {
  private http = inject(HttpClient);

  quantitativeStatistics(request: QuantitativeStatisticsRequest) {
    return this.http.post<QuantitativeStatisticsResponse>(
      `${environment.apiUrl}/statistics/all-quantitative-statistics`,
      request,
    );
  }
}
