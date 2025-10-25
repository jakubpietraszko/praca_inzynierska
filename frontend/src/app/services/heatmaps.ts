import { httpResource } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import { AllHeatmapsResponse } from '../generated-interfaces/all_heatmaps';

@Injectable({
  providedIn: 'root',
})
export class Heatmaps {
  getAllHeatmaps() {
    return httpResource<AllHeatmapsResponse>(
      () => `${environment.apiUrl}/heatmap/all-heatmaps`,
    );
  }
}
