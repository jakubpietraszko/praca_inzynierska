import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ImageService {
  private http =  inject(HttpClient);
  private apiUrl = environment.apiUrl;

  fetchImage(path: string): Observable<{image: string}> {
    return this.http.post<{image: string}>(`${this.apiUrl}/load-image`, { path });
  }
}
