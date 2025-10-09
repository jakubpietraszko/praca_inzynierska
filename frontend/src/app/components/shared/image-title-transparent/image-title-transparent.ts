import {
  ChangeDetectionStrategy,
  Component,
  input,
  signal,
} from '@angular/core';
import { MatSliderModule } from '@angular/material/slider';
import { CdkDrag, CdkDragHandle } from '@angular/cdk/drag-drop';
import { LoadModelsDatasetsImage } from '../../../directives/load-models-datasets-image';

@Component({
  selector: 'app-image-title-transparent',
  imports: [LoadModelsDatasetsImage, MatSliderModule, CdkDragHandle, CdkDrag],
  templateUrl: './image-title-transparent.html',
  styleUrl: './image-title-transparent.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
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
