import {
  afterNextRender,
  Directive,
  ElementRef,
  inject,
  input,
  Renderer2,
} from '@angular/core';

@Directive({
  selector: '[appZoomImage]',
  standalone: true,
  host: {
    '(mouseenter)': 'onMouseEnter()',
    '(mouseleave)': 'onMouseLeave()',
  },
})
export class ZoomImage {
  readonly scaleFactor = input<number>(2);

  private readonly el = inject(ElementRef);
  private readonly renderer = inject(Renderer2);

  onMouseEnter(): void {
    this.renderer.setStyle(
      this.el.nativeElement,
      'transform',
      `scale(${this.scaleFactor()})`,
    );
    this.renderer.setStyle(this.el.nativeElement, 'z-index', '1000');
  }

  onMouseLeave(): void {
    this.renderer.setStyle(this.el.nativeElement, 'transform', 'scale(1)');
    this.renderer.setStyle(this.el.nativeElement, 'z-index', 'auto');
  }
}
