import { HttpClient, HttpParams } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { DatasetBalanceResponse } from '../../../generated-interfaces/datasets_balance';
import { environment } from '../../../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class DatasetsBalance {
  private http = inject(HttpClient);

  getDatasetBalance(dataset: string): Observable<DatasetBalanceResponse> {
    const params = new HttpParams().set('dataset', dataset);
    return this.http.get<DatasetBalanceResponse>(
      `${environment.apiUrl}/datasets-balance`,
      { params },
    );
  }
}
