import { Directive, ElementRef, inject, input, OnChanges, SimpleChanges } from '@angular/core';
import { FetchImageService } from '../../services/fetch-image/fetch-image-service';
import { FetchImageResponse, FetchModelsDatasetsRequest } from '../../interfaces/fetch-image';

@Directive({
  selector: '[appLoadModelsDatasetsImage]',
  standalone: true
})
export class LoadModelsDatasetsImage implements OnChanges{
  dataset = input<string>();
  label = input<string>()
  filename = input<string>();
  model = input<string>();
  type = input<string>();

  private el = inject(ElementRef<HTMLImageElement>);
  private fetchService = inject(FetchImageService);

  constructor() { }

  ngOnChanges(changes: SimpleChanges): void {
    if(
      this.dataset() && this.label() && this.filename() && this.model() && this.type()
    ) {

      const request: FetchModelsDatasetsRequest = {
        dataset: this.dataset()!,
        label: this.label()!,
        filename: this.filename()!,
        model: this.model()!,
        type: this.type()!
      }

      this.fetchService.fetchModelsDatasetsImage(request).subscribe(
        (response: FetchImageResponse) => {
          this.el.nativeElement.src = 'data:image/png;base64,' + response.image;
        }
      )


    }
  }

}
