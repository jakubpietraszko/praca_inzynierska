import {
  Component,
  computed,
  effect,
  inject,
  input,
  OnDestroy,
  signal,
} from '@angular/core';
import { UnsupervisedActivationsService } from '../../../services/unsupervised/unsupervised-activations';
import { UnsupervisedActivationsRequest } from '../../../interfaces/unsupervised-activations';
import {
  UnsupervisedHeatmapsRequest,
  UnsupervisedHeatmapsResponse,
} from '../../../interfaces/unsupervised-heatmaps';
import { firstValueFrom, Subscription } from 'rxjs';
import { DragDropModule } from '@angular/cdk/drag-drop';
import { CommonModule } from '@angular/common';
import { LoadActivationImage } from '../../../directives/load-activation-image/load-activation-image';
import { LoadImageDirective } from "../../../directives/load-image/load-image-directive";
import { LoadModelsDatasetsImage } from "../../../directives/load-models-datasets-image/load-models-datasets-image";

interface ImageCoordinate {
  filename: string;
  coordX: number;
  coordY: number;
}

@Component({
  selector: 'app-unsupervised-heatmaps',
  imports: [DragDropModule, CommonModule, LoadModelsDatasetsImage],
  templateUrl: './unsupervised-heatmaps.html',
  styleUrl: './unsupervised-heatmaps.css',
})
export class UnsupervisedHeatmaps implements OnDestroy {
  unsupervisedService = inject(UnsupervisedActivationsService);
  imageCoordinates: ImageCoordinate[] = [];

  private subscription: Subscription = new Subscription();

  model = input<string | null>(null);
  heatmap = input<string | null>(null);
  dataset = input<string | null>(null);
  label = input<string | null>(null);
  type = input<string>('tsne');

  isDataFetched = signal<boolean>(false);

  isAllInputSet = computed(() => {
    return (
      !!this.model() &&
      !!this.dataset() &&
      !!this.label() &&
      !!this.heatmap() &&
      !!this.type()
    );
  });

  constructor() {
    effect(() => {
      if (this.isAllInputSet()) {
        this.loadHeatmaps();
      }
    });
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }

  private async loadHeatmaps(): Promise<void> {
    const request: UnsupervisedHeatmapsRequest = {
      model: this.model()!,
      heatmap: this.heatmap()!,
      dataset: this.dataset()!,
      label: this.label()!,
      type: this.type()!,
    };

    const result = await firstValueFrom(
      this.unsupervisedService.getHeatmaps(request)
    );

    this.imageCoordinates = result.filenames_coords;
    this.isDataFetched.set(true);
  }

  onDragEnded(event: any, index: number) {
    const container = event.source.element.nativeElement.parentElement;
    const containerRect = container.getBoundingClientRect();

    const imageElement = event.source.element.nativeElement;
    const imageRect = imageElement.getBoundingClientRect();

    const centerX =
      (imageRect.left + imageRect.width / 2 - containerRect.left) /
      containerRect.width;
    const centerY =
      (imageRect.top + imageRect.height / 2 - containerRect.top) /
      containerRect.height;

    this.imageCoordinates[index].coordX = centerX;
    this.imageCoordinates[index].coordY = centerY;
  }

  getCurrentCoordinates(): ImageCoordinate[] {
    return this.imageCoordinates;
  }

  resetPositions(): void {
    this.imageCoordinates = [...this.imageCoordinates];
  }
}
