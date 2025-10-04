import { Directive, effect, ElementRef, inject, input } from '@angular/core';
import { environment } from '../../environments/environment';
import { httpResource } from '@angular/common/http';
import { FetchImageResponse } from '../interfaces/fetch-image';

@Directive({
  selector: '[appLoadActivationImage]',
  standalone: true,
})
export class LoadActivationImage {
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();
  model = input.required<string>();
  image = input.required<string>();

  private el = inject(ElementRef<HTMLImageElement>);

  imageResource = httpResource<FetchImageResponse>(() => {
    const dataset = this.dataset();
    const label = this.label();
    const filename = this.filename();
    const model = this.model();
    const image = this.image();

    return {
      url: `${environment.apiUrl}/fetch-activations-image`,
      method: 'GET',
      params: {
        dataset,
        label,
        filename,
        model,
        image,
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
