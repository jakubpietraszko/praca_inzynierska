import { Component, input, signal } from '@angular/core';
import { LoadModelsDatasetsImage } from '../../../directives/load-models-datasets-image/load-models-datasets-image';
import { MatSliderModule } from '@angular/material/slider';
import { CdkDrag, CdkDragHandle } from '@angular/cdk/drag-drop';

@Component({
  selector: 'app-image-title-transparent',
  imports: [LoadModelsDatasetsImage, MatSliderModule, CdkDragHandle, CdkDrag],
  templateUrl: './image-title-transparent.html',
  styleUrl: './image-title-transparent.css',
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
