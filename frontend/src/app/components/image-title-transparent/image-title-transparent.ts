import { Component, input, ViewEncapsulation, signal } from '@angular/core';
import { LoadModelsDatasetsImage } from '../../directives/load-models-datasets-image/load-models-datasets-image';
import { MatSliderModule } from '@angular/material/slider';
import { CdkDragHandle, CdkDrag } from '@angular/cdk/drag-drop';

@Component({
  selector: 'app-image-title-transparent',
  imports: [LoadModelsDatasetsImage, MatSliderModule, CdkDragHandle, CdkDrag],
  template: `
    <div class="image-container" cdkDrag>
      <div class="slider">
        <mat-slider [min]="0" [max]="100">
          <input
            matSliderThumb
            [value]="sliderValue()"
            (input)="sliderValue.set($any($event.target).value)"
          />
        </mat-slider>
      </div>

      <img
        cdkDragHandle
        appLoadModelsDatasetsImage
        [dataset]="dataset()"
        [label]="label()"
        [filename]="filename()"
        [model]="model()"
        [type]="type()"
        [style.opacity]="sliderValue() / 100"
        class="draggable-image"
      />

      <div class="title">
        {{ title() }}
      </div>
    </div>
  `,
  styleUrl: './image-title-transparent.css',
  encapsulation: ViewEncapsulation.ShadowDom,
})
export class ImageTitleTransparent {
  title = input<string>();
  dataset = input<string>();
  label = input<string>();
  filename = input<string>();
  model = input<string>();
  type = input<string>();

  sliderValue = signal<number>(100);
}
