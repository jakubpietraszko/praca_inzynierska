import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import {
  UnsupervisedActivationsRequest,
  UnsupervisedActivationsResponse,
  GeneratingActivationsRequest,
  GeneratingActivationsResponse,
} from '../../interfaces/unsupervised-activations';
import { Observable } from 'rxjs';
import {
  UnsupervisedHeatmapsRequest,
  UnsupervisedHeatmapsResponse,
} from '../../interfaces/unsupervised-heatmaps';

@Injectable({
  providedIn: 'root',
})
export class UnsupervisedActivationsService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  getActivations(
    request: UnsupervisedActivationsRequest,
  ): Observable<UnsupervisedActivationsResponse> {
    return this.http.post<UnsupervisedActivationsResponse>(
      `${this.apiUrl}/unsupervised/activations`,
      request,
    );
  }

  generateActivations(
    request: GeneratingActivationsRequest,
  ): Observable<GeneratingActivationsResponse> {
    return this.http.post<GeneratingActivationsResponse>(
      `${this.apiUrl}/heatmap/activations`,
      request,
    );
  }

  getHeatmaps(
    request: UnsupervisedHeatmapsRequest,
  ): Observable<UnsupervisedHeatmapsResponse> {
    return this.http.post<UnsupervisedHeatmapsResponse>(
      `${this.apiUrl}/unsupervised/heatmaps`,
      request,
    );
  }
}
