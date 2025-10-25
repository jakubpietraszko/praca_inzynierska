import { httpResource } from '@angular/common/http';
import { Directive, effect, ElementRef, inject, input } from '@angular/core';
import { environment } from '../../environments/environment';
import { FetchImageResponse } from '../generated-interfaces/fetch_image';

@Directive({
  selector: '[appLoadAnyActivationsImage]',
  standalone: true,
})
export class LoadAnyActivationsImage {
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();
  model = input.required<string>();
  image = input.required<string>();
  ref = input.required<string>();

  private el = inject(ElementRef<HTMLImageElement>);

  imageResource = httpResource<FetchImageResponse>(() => {
    const dataset = this.dataset();
    const label = this.label();
    const filename = this.filename();
    const model = this.model();
    const image = this.image();
    const ref = this.ref();

    return {
      url: `${environment.apiUrl}/fetch-any-activations-image`,
      method: 'GET',
      params: {
        dataset,
        label,
        filename,
        model,
        image,
        ref,
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
