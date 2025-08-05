import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { AllModelsResponse } from '../../interfaces/all-models';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ModelsService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allModels(): Observable<AllModelsResponse> {
    return this.http.get<AllModelsResponse>(`${this.apiUrl}/all-models`);
  }
}
