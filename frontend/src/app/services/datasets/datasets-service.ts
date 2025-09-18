import { inject, Injectable, resource } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';
import { firstValueFrom, Observable } from 'rxjs';
import { AllDatasetsResponse } from '../../interfaces/all-datasets';

@Injectable({
  providedIn: 'root',
})
export class DatasetsService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allDatasets(): Observable<AllDatasetsResponse> {
    return this.http.get<AllDatasetsResponse>(`${this.apiUrl}/all-datasets`);
  }

  allDatasetsResource = resource({
    loader: () => {
      return firstValueFrom(
        this.http.get<AllDatasetsResponse>(`${this.apiUrl}/all-datasets`)
      );
    },
  });

  get datasets() {
    return this.allDatasetsResource.value;
  }

  get isLoading() {
    return this.allDatasetsResource.isLoading;
  }
}
