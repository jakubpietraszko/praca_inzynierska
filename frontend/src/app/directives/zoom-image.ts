import {
  afterNextRender,
  Directive,
  ElementRef,
  inject,
  input,
  Renderer2,
  DestroyRef,
} from '@angular/core';

@Directive({
  selector: '[appZoomImage]',
  standalone: true,
})
export class ZoomImage {
  readonly scaleFactor = input<number>(10);

  private readonly el = inject(ElementRef);
  private readonly renderer = inject(Renderer2);
  private readonly destroyRef = inject(DestroyRef);
  private isZoomed = false;
  private mouseDownListener?: () => void;
  private mouseUpListener?: () => void;

  constructor() {
    afterNextRender(() => {
      this.setupListeners();
    });
  }

  private setupListeners(): void {
    this.mouseDownListener = this.renderer.listen(
      this.el.nativeElement,
      'mousedown',
      (event: MouseEvent) => this.onMouseDown(event)
    );

    this.mouseUpListener = this.renderer.listen(
      'document',
      'mouseup',
      (event: MouseEvent) => this.onMouseUp(event)
    );

    this.destroyRef.onDestroy(() => {
      this.mouseDownListener?.();
      this.mouseUpListener?.();
    });
  }

  private onMouseDown(event: MouseEvent): void {
    if (event.button === 1) {
      event.preventDefault();
      event.stopPropagation();
      this.isZoomed = true;
      this.renderer.setStyle(this.el.nativeElement, 'position', 'fixed');
      this.renderer.setStyle(this.el.nativeElement, 'top', '50%');
      this.renderer.setStyle(this.el.nativeElement, 'left', '50%');
      this.renderer.setStyle(
        this.el.nativeElement,
        'transform',
        `translate(-50%, -50%) scale(${this.scaleFactor()})`,
      );
      this.renderer.setStyle(this.el.nativeElement, 'z-index', '100000');
      this.renderer.setStyle(this.el.nativeElement, 'pointer-events', 'none');
    }
  }

  private onMouseUp(event: MouseEvent): void {
    if (event.button === 1 && this.isZoomed) {
      this.resetZoom();
    }
  }

  private resetZoom(): void {
    this.isZoomed = false;
    this.renderer.removeStyle(this.el.nativeElement, 'position');
    this.renderer.removeStyle(this.el.nativeElement, 'top');
    this.renderer.removeStyle(this.el.nativeElement, 'left');
    this.renderer.setStyle(this.el.nativeElement, 'transform', 'none');
    this.renderer.setStyle(this.el.nativeElement, 'z-index', 'auto');
    this.renderer.removeStyle(this.el.nativeElement, 'pointer-events');
  }
}
