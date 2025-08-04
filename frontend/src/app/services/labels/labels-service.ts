import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { Observable } from 'rxjs';
import { LabelsListInterface } from '../../interfaces/labels-list-interface';
import { LabelsListRequestInterface } from '../../interfaces/labels-list-request-interface';

@Injectable({
  providedIn: 'root'
})
export class LabelsService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;

  allLabels(req: LabelsListRequestInterface): Observable<LabelsListInterface> {
    return this.http.post<LabelsListInterface>(`${this.apiUrl}/all-labels`, req);
  }
}
