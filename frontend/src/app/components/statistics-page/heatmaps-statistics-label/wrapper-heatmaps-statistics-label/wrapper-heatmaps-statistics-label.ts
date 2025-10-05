import {
  ChangeDetectionStrategy,
  Component,
  computed,
  signal,
} from '@angular/core';
import { GenericSelector } from '../../../shared/generic-selector/generic-selector';
import { ContentHeatmapsStatisticsLabel } from '../content-heatmaps-statistics-label/content-heatmaps-statistics-label';

@Component({
  selector: 'app-wrapper-heatmaps-statistics-label',
  imports: [GenericSelector, ContentHeatmapsStatisticsLabel],
  templateUrl: './wrapper-heatmaps-statistics-label.html',
  styleUrl: './wrapper-heatmaps-statistics-label.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WrapperHeatmapsStatisticsLabel {
  selectedModel = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);
  selectedHeatmap = signal<string | null>(null);
  selectedLabel = signal<string | null>(null);

  allSelected = computed(() => {
    return (
      this.selectedModel() !== null &&
      this.selectedDataset() !== null &&
      this.selectedHeatmap() !== null &&
      this.selectedLabel() !== null
    );
  });
}
