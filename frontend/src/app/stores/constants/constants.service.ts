import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ConstantModel } from './constants.interface';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class ConstantsService {
  private http = inject(HttpClient);

  getConstants(): Observable<ConstantModel> {
    return this.http.get<ConstantModel>(
      `${environment.apiUrl}/constants/xai-translations`,
    );
  }
}
