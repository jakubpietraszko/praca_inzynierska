import { Component, input, ViewEncapsulation, signal } from '@angular/core';
import { LoadModelsDatasetsImage } from '../../directives/load-models-datasets-image/load-models-datasets-image';
import { MatSliderModule } from '@angular/material/slider';

@Component({
  selector: 'app-image-title-transparent',
  imports: [LoadModelsDatasetsImage, MatSliderModule],
  template: `
    <div class="image-container" style="width: 300px; height: 300px;">
      <div class="slider" style="width: 90%;">
        <mat-slider style="width: 100%;" [min]="0" [max]="100">
          <input
            matSliderThumb
            [value]="sliderValue()"
            (input)="sliderValue.set($any($event.target).value)"
          />
        </mat-slider>
      </div>

      <img
        cdkDrag
        appLoadModelsDatasetsImage
        [dataset]="'mnist-dataset'"
        [label]="'five'"
        [filename]="'8.png'"
        [model]="'mnist'"
        [type]="'deconvolution'"
        [style.opacity]="sliderValue() / 100"
        style="width: 100%; height: 100%; object-fit: contain;"
      />

      <div class="title" style="width: 100%; text-align: center;">
        {{ title() }}
      </div>
    </div>
  `,
  styleUrl: './image-title-transparent.css',
  encapsulation: ViewEncapsulation.ShadowDom,
})
export class ImageTitleTransparent {
  title = input<string>('deconvolution');
  dataset = input<string>();
  label = input<string>();
  filename = input<string>();
  model = input<string>();
  type = input<string>();

  sliderValue = signal<number>(100);
}
