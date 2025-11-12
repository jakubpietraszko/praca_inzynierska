import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import {
  ModelDatasetI,
  Response,
} from '../../generated-interfaces/numerator_denominator';
import { environment } from '../../../environments/environment';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

@Injectable()
export class GeneratorDirectory {
  private http = inject(HttpClient);

  private destroyRef = inject(DestroyRef);

  public directoryCreated = signal(false);

  createDirectory(request: ModelDatasetI): void {
    this.directoryCreated.set(false);
    this.http
      .put<Response>(`${environment.apiUrl}/models-datasets`, request)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => {
          this.directoryCreated.set(true);
        },
      });
  }
}
