import { HttpClient, HttpParams } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface ModelDatasetI {
  model: string;
  dataset: string;
}

export interface ModelDatasetHeatmapI extends ModelDatasetI {
  heatmap: string;
}

export interface NumeratorDenominatorI {
  numerator: number;
  denominator: number;
}

@Injectable({
  providedIn: 'root',
})
export class HeatmapsGenerator {
  private http = inject(HttpClient);

  createDatasetsModelsDir(
    request: ModelDatasetI
  ): Observable<{ success: boolean }> {
    return this.http.put<{ success: boolean }>(
      `${environment.apiUrl}/models-datasets`,
      request
    );
  }

  getNumeratorDenumerator(
    request: ModelDatasetHeatmapI
  ): Observable<NumeratorDenominatorI> {
    const params = new HttpParams()
      .set('model', request.model)
      .set('dataset', request.dataset)
      .set('heatmap', request.heatmap);

    return this.http.get<NumeratorDenominatorI>(
      `${environment.apiUrl}/numerator-denominator`,
      { params }
    );
  }
}
