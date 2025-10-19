import { Injectable } from '@angular/core';
import { AllHeatmapsResponse } from '../interfaces/all-heatmaps';
import { httpResource } from '@angular/common/http';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class AllHeatmaps {
  public heatmaps = httpResource<AllHeatmapsResponse>(
    () => `${environment.apiUrl}/heatmap/all-heatmaps`,
  );
}
