import { HttpClient } from '@angular/common/http';
import { DestroyRef, inject, Injectable, signal } from '@angular/core';
import {
  ProbaForImageRequest,
  ProbaForImageResponse,
} from '../../../generated-interfaces/proba_for_image';
import { environment } from '../../../../environments/environment';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

@Injectable()
export class ProbaImage {
  private readonly http = inject(HttpClient);

  private readonly destroyRef = inject(DestroyRef);

  public readonly response = signal<ProbaForImageResponse | null>(null);

  public fetchProbaForImage(request: ProbaForImageRequest): void {
    this.response.set(null);

    this.http
      .post<ProbaForImageResponse>(
        `${environment.apiUrl}/statistics/proba-for-image`,
        request,
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((response: ProbaForImageResponse) => {
        this.response.set(response);
      });
  }
}
