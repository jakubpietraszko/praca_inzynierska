import { Directive, effect, ElementRef, inject, input } from '@angular/core';
import { FetchImageResponse } from '../interfaces/fetch-image';
import { httpResource } from '@angular/common/http';
import { environment } from '../../environments/environment';

@Directive({
  selector: '[appLoadImage]',
  standalone: true,
})
export class LoadImage {
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();

  private el = inject(ElementRef<HTMLImageElement>);

  imageResource = httpResource<FetchImageResponse>(() => {
    const dataset = this.dataset();
    const label = this.label();
    const filename = this.filename();

    return {
      url: `${environment.apiUrl}/fetch-image`,
      method: 'GET',
      params: {
        dataset,
        label,
        filename,
      },
    };
  });

  constructor() {
    effect(() => {
      const value = this.imageResource.value();
      if (value) {
        this.el.nativeElement.src = 'data:image/png;base64,' + value.image;
      }
    });
  }
}
