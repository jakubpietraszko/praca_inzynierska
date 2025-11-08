import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import { environment } from '../../../../environments/environment';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  HeatmapStatisticsLabelManyRequest,
  HeatmapStatisticsLabelManyResponse,
} from '../../../generated-interfaces/heatmap';

@Injectable()
export class Scores {
  private readonly http = inject(HttpClient);

  private readonly destroyRef = inject(DestroyRef);

  public readonly response = signal<HeatmapStatisticsLabelManyResponse | null>(
    null,
  );

  public fetchShapesStatisticsLabel(
    request: HeatmapStatisticsLabelManyRequest,
  ): void {
    this.response.set(null);

    this.http
      .post<HeatmapStatisticsLabelManyResponse>(
        `${environment.apiUrl}/heatmap/heatmaps-statistics-label-many`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((response: HeatmapStatisticsLabelManyResponse) => {
        this.response.set(response);
      });
  }
}
