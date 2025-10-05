import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ConfigModel } from './config.interface';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class ConfigService {
  private http = inject(HttpClient);

  getConfig(): Observable<ConfigModel> {
    return this.http.get<ConfigModel>(`${environment.apiUrl}/config`);
  }

  saveConfig(config: ConfigModel): Observable<any> {
    return this.http.put<any>(`${environment.apiUrl}/config`, config);
  }
}
