import { HttpClient } from '@angular/common/http';
import { inject, Injectable, resource } from '@angular/core';
import { environment } from '../../../environments/environment';
import { AllModelsResponse } from '../../interfaces/all-models';
import { firstValueFrom, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ModelsService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allModels(): Observable<AllModelsResponse> {
    return this.http.get<AllModelsResponse>(`${this.apiUrl}/all-models`);
  }

  allModelsResource = resource({
    loader: () => {
      return firstValueFrom(
        this.http.get<AllModelsResponse>(`${this.apiUrl}/all-models`)
      );
    },
  });

  get models() {
    return this.allModelsResource.value;
  }

  get isLoading() {
    return this.allModelsResource.isLoading;
  }
}
