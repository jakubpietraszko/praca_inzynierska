import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import { environment } from '../../../environments/environment';
import {
  ModelDatasetStatisticsRequest,
  ModelDatasetStatisticsResponse,
} from '../../generated-interfaces/model_dataset_statistics';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

@Injectable()
export class GeneratorStatistics {
  private http = inject(HttpClient);
  private destroyRef = inject(DestroyRef);

  public response = signal<ModelDatasetStatisticsResponse | null>(null);

  public fetchStatistics(request: ModelDatasetStatisticsRequest): void {
    this.response.set(null);
    this.http
      .post<ModelDatasetStatisticsResponse>(
        `${environment.apiUrl}/statistics/model-dataset-statistics`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((data) => {
        this.response.set(data);
      });
  }
}
