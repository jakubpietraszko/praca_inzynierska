import {
  ChangeDetectionStrategy,
  Component,
  effect,
  inject,
  input,
  signal,
} from '@angular/core';
import {
  CompareHeatmapsRequest,
  CompareHeatmapsResponse,
  GeneratorCompareHeatmaps,
} from '../generator-compare-heatmaps';
import { ImageTitleTransparent } from '../../../shared/image-title-transparent/image-title-transparent';
import { AllHeatmaps } from '../../../../services/all-heatmaps';

@Component({
  selector: 'app-content-compare-heatmaps',
  imports: [ImageTitleTransparent],
  templateUrl: './content-compare-heatmaps.html',
  styleUrl: './content-compare-heatmaps.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentCompareHeatmaps {
  model = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
  filename = input.required<string>();

  response = signal<CompareHeatmapsResponse | null>(null);

  compareHeatmapsService = inject(GeneratorCompareHeatmaps);

  allHeatmapsService = inject(AllHeatmaps);

  constructor() {
    effect((onCleanup) => {
      this.response.set(null);
      const request: CompareHeatmapsRequest = {
        model: this.model(),
        dataset: this.dataset(),
        images: [
          {
            label: this.label(),
            filename: this.filename(),
            target: null,
          },
        ],
        overwrite: false,
      };

      const subscription = this.compareHeatmapsService
        .compareHeatmaps(request)
        .subscribe((response: CompareHeatmapsResponse) => {
          this.response.set(response);
        });

      onCleanup(() => {
        subscription.unsubscribe();
      });
    });
  }
}
