import { Injectable } from '@angular/core';
import { AllDatasetsResponse } from '../interfaces/all-datasets';
import { httpResource } from '@angular/common/http';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class AllDatasets {
  public datasets = httpResource<AllDatasetsResponse>(
    () => `${environment.apiUrl}/all-datasets`,
  );
}
