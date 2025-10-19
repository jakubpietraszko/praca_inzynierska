import { httpResource } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { AllModelsResponse } from '../interfaces/all-models';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class AllModels {
  public models = httpResource<AllModelsResponse>(
    () => `${environment.apiUrl}/all-models`,
  );
}
