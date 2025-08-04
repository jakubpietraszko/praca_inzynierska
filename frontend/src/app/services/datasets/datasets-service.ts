import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';
import { Observable } from 'rxjs';
import { DatasetsListInterface } from '../../interfaces/datasets-list-interface';

@Injectable({
  providedIn: 'root'
})
export class DatasetsService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allDatasets(): Observable<DatasetsListInterface> {
    return this.http.get<DatasetsListInterface>(`${this.apiUrl}/all-datasets`);
  }
}
