import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { ConfigModel } from './config.interface';

@Injectable({
  providedIn: 'root',
})
export class ConfigService {
  private apiUrl = environment.apiUrl;

  private http = inject(HttpClient);

  getConfig(): Observable<ConfigModel> {
    return this.http.get<ConfigModel>(`${this.apiUrl}/config`);
  }

  saveConfig(config: ConfigModel): Observable<any> {
    return this.http.put<any>(`${this.apiUrl}/config`, config);
  }
}
