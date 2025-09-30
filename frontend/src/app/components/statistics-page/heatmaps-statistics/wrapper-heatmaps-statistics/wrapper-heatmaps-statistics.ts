import {
  ChangeDetectionStrategy,
  Component,
  computed,
  signal,
} from '@angular/core';
import { GenericSelector } from '../../../shared/generic-selector/generic-selector';
import { ContentHeatmapsStatistics } from '../content-heatmaps-statistics/content-heatmaps-statistics';

@Component({
  selector: 'app-wrapper-heatmaps-statistics',
  imports: [GenericSelector, ContentHeatmapsStatistics],
  templateUrl: './wrapper-heatmaps-statistics.html',
  styleUrl: './wrapper-heatmaps-statistics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WrapperHeatmapsStatistics {
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
