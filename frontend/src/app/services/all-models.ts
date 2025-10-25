import { httpResource } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import { AllModelsResponse } from '../generated-interfaces/all_models';

@Injectable({
  providedIn: 'root',
})
export class AllModels {
  public models = httpResource<AllModelsResponse>(
    () => `${environment.apiUrl}/all-models`,
  );
}
