import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import { environment } from '../../../../environments/environment';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  ShapeMetricsDatasetRequest,
  ShapeMetricsDatasetResponse,
} from '../../../generated-interfaces/shapes';

@Injectable()
export class Shapes {
  private readonly http = inject(HttpClient);

  private readonly destroyRef = inject(DestroyRef);

  public readonly response = signal<ShapeMetricsDatasetResponse | null>(null);

  public fetchShapesStatisticsDataset(
    request: ShapeMetricsDatasetRequest,
  ): void {
    this.response.set(null);

    this.http
      .post<ShapeMetricsDatasetResponse>(
        `${environment.apiUrl}/heatmap/heatmaps-statistics-dataset-shapes`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((response: ShapeMetricsDatasetResponse) => {
        this.response.set(response);
      });
  }
}
