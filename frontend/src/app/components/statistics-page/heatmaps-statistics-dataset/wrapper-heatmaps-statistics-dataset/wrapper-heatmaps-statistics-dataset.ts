import {
  ChangeDetectionStrategy,
  Component,
  computed,
  signal,
} from '@angular/core';
import { GenericSelector } from '../../../shared/generic-selector/generic-selector';
import { ContentHeatmapsStatisticsDataset } from '../content-heatmaps-statistics-dataset/content-heatmaps-statistics-dataset';

@Component({
  selector: 'app-wrapper-heatmaps-statistics-dataset',
  imports: [GenericSelector, ContentHeatmapsStatisticsDataset],
  templateUrl: './wrapper-heatmaps-statistics-dataset.html',
  styleUrl: './wrapper-heatmaps-statistics-dataset.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WrapperHeatmapsStatisticsDataset {
  selectedModel = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);
  selectedHeatmap = signal<string | null>(null);

  allSelected = computed(() => {
    return (
      this.selectedModel() !== null &&
      this.selectedDataset() !== null &&
      this.selectedHeatmap() !== null
    );
  });
}
