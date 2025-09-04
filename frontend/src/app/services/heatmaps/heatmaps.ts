import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { Observable } from 'rxjs';
import { AllHeatmapsResponse } from '../../interfaces/all-heatmaps';

@Injectable({
  providedIn: 'root',
})
export class Heatmaps {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allHeatmaps(): Observable<AllHeatmapsResponse> {
    return this.http.get<AllHeatmapsResponse>(`${this.apiUrl}/all-heatmaps`);
  }
}
