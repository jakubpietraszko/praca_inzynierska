import {
  ChangeDetectionStrategy,
  Component,
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
import { DragDropModule } from '@angular/cdk/drag-drop';
import { LoadModelsDatasetsImage } from '../../../../directives/load-models-datasets-image/load-models-datasets-image';

@Component({
  selector: 'app-content-unsupervised-heatmaps',
  imports: [DragDropModule, LoadModelsDatasetsImage],
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
            })
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
