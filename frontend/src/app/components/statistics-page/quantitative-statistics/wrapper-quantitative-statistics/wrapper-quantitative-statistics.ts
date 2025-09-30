import {
  ChangeDetectionStrategy,
  Component,
  computed,
  signal,
} from '@angular/core';
import { GenericSelector } from '../../../shared/generic-selector/generic-selector';
import { ContentQuantitativeStatistics } from '../content-quantitative-statistics/content-quantitative-statistics';

@Component({
  selector: 'app-wrapper-quantitative-statistics',
  imports: [GenericSelector, ContentQuantitativeStatistics],
  templateUrl: './wrapper-quantitative-statistics.html',
  styleUrl: './wrapper-quantitative-statistics.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WrapperQuantitativeStatistics {
  selectedModel = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);

  allSelected = computed(() => {
    return this.selectedDataset() && this.selectedModel();
  });
}
