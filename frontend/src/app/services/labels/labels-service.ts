import { HttpClient } from '@angular/common/http';
import { inject, Injectable, resource } from '@angular/core';
import { environment } from '../../../environments/environment';
import { Observable } from 'rxjs';
import { AllLabelsRequest, AllLabelsResponse } from '../../interfaces/all-labels';

@Injectable({
  providedIn: 'root'
})
export class LabelsService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allLabels(request: AllLabelsRequest): Observable<AllLabelsResponse> {
    return this.http.post<AllLabelsResponse>(`${this.apiUrl}/all-labels`, request);
  }

  

}
