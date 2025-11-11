import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import {
  UnsupervisedLabelRequest,
  UnsupervisedLabelResponse,
} from '../../../generated-interfaces/unsupervised_label';
import { environment } from '../../../../environments/environment';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

@Injectable()
export class GeneratorUnsupervisedLabel {
  private http = inject(HttpClient);

  private destroyRef = inject(DestroyRef);

  public readonly response = signal<UnsupervisedLabelResponse | null>(null);

  public fetchUnsupervisedLabel(request: UnsupervisedLabelRequest): void {
    this.response.set(null);

    this.http
      .post<UnsupervisedLabelResponse>(
        `${environment.apiUrl}/unsupervised/label`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((response: UnsupervisedLabelResponse) => {
        this.response.set(response);
      });
  }
}
