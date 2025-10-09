import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import {
  AllImagesRequest,
  AllImagesResponse,
} from '../../interfaces/all-images';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ImagesService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allImages(request: AllImagesRequest): Observable<AllImagesResponse> {
    return this.http.post<AllImagesResponse>(
      `${this.apiUrl}/all-images`,
      request,
    );
  }
}
