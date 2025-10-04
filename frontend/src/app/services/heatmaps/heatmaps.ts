import { HttpClient } from '@angular/common/http';
import { inject, Injectable, resource } from '@angular/core';
import { environment } from '../../../environments/environment';
import { firstValueFrom, Observable } from 'rxjs';
import {
  HeatmapResponse,
  HeatmapRequest,
  AllHeatmapsResponse,
} from '../../interfaces/all-heatmaps';

@Injectable({
  providedIn: 'root',
})
export class HeatmapsService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allHeatmaps(): Observable<AllHeatmapsResponse> {
    return this.http.get<AllHeatmapsResponse>(
      `${this.apiUrl}/heatmap/all-heatmaps`
    );
  }

  generateHeatmap(request: HeatmapRequest): Observable<HeatmapResponse> {
    return this.http.post<HeatmapResponse>(
      `${this.apiUrl}/heatmap/all-heatmaps`,
      request
    );
  }

  allHeatmapsResource = resource({
    loader: () => {
      return firstValueFrom(
        this.http.get<AllHeatmapsResponse>(
          `${this.apiUrl}/heatmap/all-heatmaps`
        )
      );
    },
  });

  get heatmaps() {
    return this.allHeatmapsResource.value;
  }

  get isLoading() {
    return this.allHeatmapsResource.isLoading;
  }
}
