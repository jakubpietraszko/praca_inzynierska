import { Directive, effect, ElementRef, inject, input } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { FetchImageResponse } from '../generated-interfaces/fetch_image';

@Directive({
  selector: '[appLoadFourier]',
  standalone: true,
})
export class LoadFourier {
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();
  model = input.required<string>();
  type = input.required<string>();

  private el = inject(ElementRef<HTMLImageElement>);

  imageResource = httpResource<FetchImageResponse>(() => {
    const dataset = this.dataset();
    const label = this.label();
    const filename = this.filename();
    const model = this.model();
    const type = this.type();

    return {
      url: `${environment.apiUrl}/fetch-models-datasets-fourier-image`,
      method: 'GET',
      params: {
        dataset,
        label,
        filename,
        model,
        type,
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
