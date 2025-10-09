import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import {
  ModelLayersRequest,
  ModelLayersResponse,
} from '../../interfaces/model-layers';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ModelLayersService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  modelLayers(request: ModelLayersRequest): Observable<ModelLayersResponse> {
    return this.http.post<ModelLayersResponse>(
      `${this.apiUrl}/model-layers`,
      request,
    );
  }
}
