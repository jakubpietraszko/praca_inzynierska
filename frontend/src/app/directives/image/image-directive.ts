import { Directive, effect, ElementRef, inject, input } from '@angular/core';
import { ImageService } from '../../services/image/image-service';

@Directive({
  selector: '[appImageDirective]',
  standalone: true
})
export class ImageDirective {
  private el = inject(ElementRef<HTMLImageElement>);
  private imageService = inject(ImageService);

  imagePath = input<string>('');

  constructor() { 
    effect(() => {
      this.loadImage(this.imagePath());
    });
  }

  private loadImage(path: string): void {
    if(!path) {
      console.error('Image path is required');
      return;
    }

    this.imageService.fetchImage(path)
      .subscribe({
        next: (response) => {
          this.el.nativeElement.src = 'data:image/png;base64,' + response.image
        },
        error: (error) => {
          console.error('Error fetching image:', error);
        }
      })
  }

}
