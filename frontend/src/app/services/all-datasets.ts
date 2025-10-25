import { Injectable } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { AllDatasetsResponse } from '../generated-interfaces/all_datasets';

@Injectable({
  providedIn: 'root',
})
export class AllDatasets {
  public datasets = httpResource<AllDatasetsResponse>(
    () => `${environment.apiUrl}/all-datasets`,
  );
}
