import {
  Directive,
  ElementRef,
  inject,
  Input,
  OnChanges,
  SimpleChanges,
} from '@angular/core';
import { FetchImageService } from '../../services/fetch-image/fetch-image-service';
import { FetchActivationRequest } from '../../interfaces/fetch-image';

@Directive({
  selector: '[appLoadActivationImage]',
  standalone: true,
})
export class LoadActivationImage implements OnChanges {
  @Input() dataset!: string;
  @Input() label!: string;
  @Input() filename!: string;
  @Input() model!: string;
  @Input() image!: string;

  private el = inject(ElementRef<HTMLImageElement>);
  private fetchService = inject(FetchImageService);

  constructor() {}

  ngOnChanges(changes: SimpleChanges): void {
    if (
      this.dataset &&
      this.label &&
      this.filename &&
      this.model &&
      this.image
    ) {
      const request: FetchActivationRequest = {
        dataset: this.dataset,
        label: this.label,
        filename: this.filename,
        model: this.model,
        image: this.image,
      };

      this.fetchService.fetchActivationImage(request).subscribe({
        next: (response) => {
          this.el.nativeElement.src = 'data:image/png;base64,' + response.image;
        },
        error: (error) => {
          console.error('Error loading activation image:', error);
          console.error('Request was:', request);
          // Set a placeholder or error image
          this.el.nativeElement.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjYwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjRkZGIiBzdHJva2U9IiNEREQiLz4KPHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHRleHQgeD0iMzAiIHk9IjMwIiBmaWxsPSIjOTk5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkb21pbmFudC1iYXNlbGluZT0iY2VudHJhbCIgZm9udC1zaXplPSIxMiI+RXJyb3I8L3RleHQ+Cjwvc3ZnPgo=';
        }
      });
    }
  }
}
