import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
  OnDestroy,
  signal,
} from '@angular/core';
import {
  GeneratorUnsupervisedHeatmaps,
  UnsupervisedHeatmapsRequest,
  UnsupervisedHeatmapsResponse,
} from '../generator-unsupervised-heatmaps';
import { Subscription } from 'rxjs';
import { LoadModelsDatasetsImage } from '../../../../directives/load-models-datasets-image';
import {
  GenericUnsupervised,
  UnsupervisedCoordinate,
} from '../../../shared/generic-unsupervised/generic-unsupervised';
import { ZoomImage } from '../../../../directives/zoom-image';

@Component({
  selector: 'app-content-unsupervised-heatmaps',
  imports: [LoadModelsDatasetsImage, GenericUnsupervised, ZoomImage],
  templateUrl: './content-unsupervised-heatmaps.html',
  styleUrl: './content-unsupervised-heatmaps.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentUnsupervisedHeatmaps implements OnDestroy {
  model = input.required<string>();
  heatmap = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
  type = signal<string>('tsne');

  response = signal<UnsupervisedHeatmapsResponse | null>(null);

  coordinates = computed<UnsupervisedCoordinate[]>(() => {
    const res = this.response();
    if (!res) return [];
    return res.filenames_coords.map(fc => ({
      coordX: fc.coordX,
      coordY: fc.coordY,
      label: this.label(),
      filename: fc.filename,
    }));
  });

  private subscription = new Subscription();

  unsupervisedHeatmapsService = inject(GeneratorUnsupervisedHeatmaps);

  constructor() {
    effect(() => {
      if (this.model() && this.dataset() && this.label() && this.heatmap()) {
        this.response.set(null);
        const request: UnsupervisedHeatmapsRequest = {
          model: this.model(),
          dataset: this.dataset(),
          label: this.label(),
          heatmap: this.heatmap(),
          type: this.type(),
        };

        if (this.subscription) {
          this.subscription.unsubscribe();
          this.subscription = new Subscription();
        }

        this.subscription.add(
          this.unsupervisedHeatmapsService
            .unsupervisedHeatmaps(request)
            .subscribe((response: UnsupervisedHeatmapsResponse) => {
              this.response.set(response);
            }),
        );
      }
    });
  }

  ngOnDestroy(): void {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
  }
}
