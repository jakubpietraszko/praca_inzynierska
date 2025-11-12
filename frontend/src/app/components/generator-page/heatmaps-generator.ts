import { HttpClient, HttpParams } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class HeatmapsGenerator {
  private http = inject(HttpClient);

  generateAllHeatmapsByType(
    model: string,
    dataset: string,
    label: string,
    type: string,
  ): Observable<{ success: boolean }> {
    const params = new HttpParams()
      .set('model', model)
      .set('dataset', dataset)
      .set('label', label)
      .set('type', type);

    return this.http.get<{ success: boolean }>(
      `${environment.apiUrl}/heatmap/all-heatmaps-type`,
      { params },
    );
  }
}
