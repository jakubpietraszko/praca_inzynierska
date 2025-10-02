import {
  ChangeDetectionStrategy,
  Component,
  computed,
  signal,
} from '@angular/core';
import { GenericSelector } from '../../../shared/generic-selector/generic-selector';
import { ContentHeatmapsStatisticsImage } from '../content-heatmaps-statistics-image/content-heatmaps-statistics-image';

@Component({
  selector: 'app-wrapper-heatmaps-statistics-image',
  imports: [GenericSelector, ContentHeatmapsStatisticsImage],
  templateUrl: './wrapper-heatmaps-statistics-image.html',
  styleUrl: './wrapper-heatmaps-statistics-image.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WrapperHeatmapsStatisticsImage {
  selectedModel = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);
  selectedHeatmap = signal<string | null>(null);
  selectedLabel = signal<string | null>(null);
  selectedImage = signal<string | null>(null);

  allSelected = computed(() => {
    return (
      this.selectedModel() !== null &&
      this.selectedDataset() !== null &&
      this.selectedHeatmap() !== null &&
      this.selectedLabel() !== null &&
      this.selectedImage() !== null
    );
  });
}
