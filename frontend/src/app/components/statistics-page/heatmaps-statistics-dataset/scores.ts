import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import { environment } from '../../../../environments/environment';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  HeatmapStatisticsDatasetManyRequest,
  HeatmapStatisticsDatasetManyResponse,
} from '../../../generated-interfaces/heatmap';

@Injectable()
export class Scores {
  private readonly http = inject(HttpClient);

  private readonly destroyRef = inject(DestroyRef);

  public readonly response = signal<HeatmapStatisticsDatasetManyResponse | null>(
    null,
  );

  public fetchScoresStatisticsDataset(
    request: HeatmapStatisticsDatasetManyRequest,
  ): void {
    this.response.set(null);

    this.http
      .post<HeatmapStatisticsDatasetManyResponse>(
        `${environment.apiUrl}/heatmap/heatmaps-statistics-dataset-many`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((response: HeatmapStatisticsDatasetManyResponse) => {
        this.response.set(response);
      });
  }
}
