import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import {
  UnsupervisedDatasetRequest,
  UnsupervisedDatasetResponse,
} from '../../../generated-interfaces/unsupervised_dataset';
import { environment } from '../../../../environments/environment';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

@Injectable()
export class GeneratorUnsupervisedDataset {
  private http = inject(HttpClient);

  private destroyRef = inject(DestroyRef);

  public readonly response = signal<UnsupervisedDatasetResponse| null>(null);

  public fetchUnsupervisedDataset(request: UnsupervisedDatasetRequest): void {
    this.response.set(null);

    this.http
      .post<UnsupervisedDatasetResponse>(
        `${environment.apiUrl}/unsupervised/dataset`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((response: UnsupervisedDatasetResponse) => {
        this.response.set(response);
      });
  }
}
