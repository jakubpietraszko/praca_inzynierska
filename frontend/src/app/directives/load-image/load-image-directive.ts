import { Directive, ElementRef, inject, Input, OnChanges, SimpleChanges } from '@angular/core';
import { FetchImageService } from '../../services/fetch-image/fetch-image-service';
import { FetchImageRequest, FetchImageResponse } from '../../interfaces/fetch-image';

@Directive({
  selector: '[appLoadImageDirective]',
  standalone: true
})
export class LoadImageDirective implements OnChanges{
  @Input() dataset!: string;
  @Input() label!: string;
  @Input() filename!: string;

  private el = inject(ElementRef<HTMLImageElement>);
  private fetchService = inject(FetchImageService);

  constructor() { }

  ngOnChanges(changes: SimpleChanges): void {
    if(this.dataset && this.label && this.filename) {
      const request: FetchImageRequest = {
        dataset: this.dataset,
        label: this.label,
        filename: this.filename
      };

      this.fetchService.fetchImage(request).subscribe({
        next: (response: FetchImageResponse) => {
          this.el.nativeElement.src = 'data:image/png;base64,' + response.image;
        }
      });
    }
  }

}
