import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';
import { GenericDatasetSelector } from '../shared/generic-dataset-selector/generic-dataset-selector';
import { GenericLabelSelector } from '../shared/generic-label-selector/generic-label-selector';
import { MatIconModule } from '@angular/material/icon';
import { DialogCompareHeatmaps } from './compare-heatmaps/dialog-compare-heatmaps';
import { DialogUnsupervisedHeatmaps } from './unsupervised-heatmaps/dialog-unsupervised-heatmaps';
import { DialogUnsupervisedActivations } from './unsupervised-activations/dialog-unsupervised-activations';
import { httpResource } from '@angular/common/http';
import { AllImagesResponse } from '../../interfaces/all-images';
import { environment } from '../../../environments/environment';
import { constants } from '../../../constants';
import { MatPaginatorModule } from '@angular/material/paginator';
import { LoadImageDirective } from '../../directives/load-image/load-image-directive';
import { MatGridListModule } from '@angular/material/grid-list';

@Component({
  selector: 'app-datasets',
  imports: [
    GenericDatasetSelector,
    GenericLabelSelector,
    MatIconModule,
    MatPaginatorModule,
    LoadImageDirective,
    MatGridListModule,
  ],
  templateUrl: './datasets.html',
  styleUrl: './datasets.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Datasets {
  imagesPerPage = constants.imagesPerPage;
  imagesPerRow = constants.imagesPerRow;

  compareHeatmapsDialog = inject(DialogCompareHeatmaps);
  unsupervisedActivationsDialog = inject(DialogUnsupervisedActivations);
  unsupervisedHeatmapsDialog = inject(DialogUnsupervisedHeatmaps);

  currentDataset = signal<string | null>(null);
  currentLabel = signal<string | null>(null);

  currentPage = signal<number | null>(null);
  totalImages = signal<number | null>(null);
  images = httpResource<AllImagesResponse>(() => {
    const dataset = this.currentDataset();
    const label = this.currentLabel();
    if (!dataset || !label) return undefined;
    return {
      url: `${environment.apiUrl}/all-images`,
      method: 'GET',
      params: {
        dataset,
        label,
      },
    };
  });

  paginatedImages = computed(() => {
    if (!this.images || this.images.isLoading()) {
      return [];
    }

    const images = this.images.value()!.images;
    const page = this.currentPage() ?? 0;
    const startIndex = page * constants.imagesPerPage;
    const endIndex = startIndex + constants.imagesPerPage;
    return images.slice(startIndex, endIndex);
  });

  constructor() {}

  onUnsupervisedHeatmapsOpen(dataset: string, label: string) {
    this.unsupervisedHeatmapsDialog.open(dataset, label);
  }

  onCompareHeatmapsOpen(dataset: string, label: string, filename: string) {
    this.compareHeatmapsDialog.open(dataset, label, filename);
  }

  onUnsupervisedActivationsOpen(dataset: string, label: string, filename: string) {
    this.unsupervisedActivationsDialog.open(dataset, label, filename);
  }

  onPageChange(event: any) {
    this.currentPage.set(event.pageIndex);
  }

  get gridImageSize() {
    return `calc((100% - ${(this.imagesPerRow - 1) * 8}px) / ${
      this.imagesPerRow
    })`;
  }
}
