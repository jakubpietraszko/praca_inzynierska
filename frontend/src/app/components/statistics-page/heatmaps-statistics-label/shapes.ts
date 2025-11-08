import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import { environment } from '../../../../environments/environment';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  ShapeMetricsLabelRequest,
  ShapeMetricsLabelResponse,
} from '../../../generated-interfaces/shapes';

@Injectable()
export class Shapes {
  private readonly http = inject(HttpClient);

  private readonly destroyRef = inject(DestroyRef);

  public readonly response = signal<ShapeMetricsLabelResponse | null>(null);

  public fetchShapesStatisticsLabel(request: ShapeMetricsLabelRequest): void {
    this.response.set(null);

    this.http
      .post<ShapeMetricsLabelResponse>(
        `${environment.apiUrl}/heatmap/heatmaps-statistics-label-shapes`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((response: ShapeMetricsLabelResponse) => {
        this.response.set(response);
      });
  }
}
