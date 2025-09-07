import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { Observable } from 'rxjs';
import { LabelFilenameTarget, HeatmapResponse, HeatmapRequest, AllHeatmapsResponse } from '../../interfaces/all-heatmaps';

@Injectable({
  providedIn: 'root',
})
export class Heatmaps {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allHeatmaps(): Observable<AllHeatmapsResponse> {
    return this.http.get<AllHeatmapsResponse>(`${this.apiUrl}/heatmap/all-heatmaps`);
  }


  generateHeatmap(request: HeatmapRequest): Observable<HeatmapResponse> {
    return this.http.post<HeatmapResponse>(`${this.apiUrl}/heatmap/all-heatmaps`, request);
  }

}
