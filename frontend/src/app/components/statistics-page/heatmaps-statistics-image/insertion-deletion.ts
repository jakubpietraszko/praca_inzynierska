import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  InsertionDeletionRequest,
  InsertionDeletionResponse,
} from '../../../generated-interfaces/insertion_deletion';
import { environment } from '../../../../environments/environment';

@Injectable()
export class InsertionDeletion {
  private readonly http = inject(HttpClient);

  private readonly destroyRef = inject(DestroyRef);

  public readonly response = signal<InsertionDeletionResponse | null>(null);

  public fetchInsertionDeletion(request: InsertionDeletionRequest): void {
    this.response.set(null);

    this.http
      .post<InsertionDeletionResponse>(
        `${environment.apiUrl}/heatmap/heatmaps-statistics-image-insertion-deletion`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((response: InsertionDeletionResponse) => {
        this.response.set(response);
      });
  }
}
