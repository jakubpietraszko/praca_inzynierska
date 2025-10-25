import { Injectable } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { AllHeatmapsResponse } from '../generated-interfaces/all_heatmaps';

@Injectable({
  providedIn: 'root',
})
export class AllHeatmaps {
  public heatmaps = httpResource<AllHeatmapsResponse>(
    () => `${environment.apiUrl}/heatmap/all-heatmaps`,
  );
}
