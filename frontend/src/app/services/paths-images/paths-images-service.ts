import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { PathsImagesRequestInterface } from '../../interfaces/paths-images-request-interface';
import { Observable } from 'rxjs';
import { PathsImagesListInterface } from '../../interfaces/paths-images-list-interface';

@Injectable({
  providedIn: 'root'
})
export class PathsImagesService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allPathsImages(req: PathsImagesRequestInterface): Observable<PathsImagesListInterface> {
    return this.http.post<PathsImagesListInterface>(`${this.apiUrl}/all-paths-images`, req);
  }
}
