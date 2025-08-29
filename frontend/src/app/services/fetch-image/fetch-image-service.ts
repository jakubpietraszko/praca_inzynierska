import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { FetchImageRequest, FetchImageResponse, FetchActivationRequest } from '../../interfaces/fetch-image';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class FetchImageService {
  private http = inject(HttpClient)
  private apiUrl = environment.apiUrl;

  fetchImage(request: FetchImageRequest): Observable<FetchImageResponse> {
    return this.http.post<FetchImageResponse>(`${this.apiUrl}/fetch-image`, request);
  }

  fetchActivationImage(request: FetchActivationRequest): Observable<FetchImageResponse> {
    return this.http.post<FetchImageResponse>(`${this.apiUrl}/fetch-activations-image`, request);
  }
}
