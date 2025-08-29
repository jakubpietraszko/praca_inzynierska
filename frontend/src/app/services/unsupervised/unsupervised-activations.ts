import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { UnsupervisedActivationsRequest, UnsupervisedActivationsResponse, GeneratingActivationsRequest, GeneratingActivationsResponse, LabelFilenamesCoordinates} from '../../interfaces/unsupervised-activations';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class UnsupervisedActivationsService {

  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  getActivations(request: UnsupervisedActivationsRequest): Observable<UnsupervisedActivationsResponse> {
    return this.http.post<UnsupervisedActivationsResponse>(`${this.apiUrl}/unsupervised/activations`, request);
  }

  generateActivations(request: GeneratingActivationsRequest): Observable<GeneratingActivationsResponse> {
    return this.http.post<GeneratingActivationsResponse>(`${this.apiUrl}/heatmap/activations`, request);
  }

}
