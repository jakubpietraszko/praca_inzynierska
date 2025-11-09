import {
  ChangeDetectionStrategy,
  Component,
  contentChild,
  effect,
  model,
  TemplateRef,
  inject,
} from '@angular/core';
import { MatTabsModule } from '@angular/material/tabs';
import { CommonModule } from '@angular/common';
import { AllDatasets } from '../../../services/all-datasets';

@Component({
  selector: 'app-generic-dataset-selector',
  imports: [MatTabsModule, CommonModule],
  templateUrl: './generic-dataset-selector.html',
  styleUrl: './generic-dataset-selector.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericDatasetSelector {
  private readonly allDatasets = inject(AllDatasets);

  postfixButtonTemplate = contentChild('postfixButton', { read: TemplateRef });

  selectedDataset = model<string | null>(null);

  datasets = this.allDatasets.datasets;

  constructor() {
    effect(() => {
      const datasetsValue = this.datasets.value();
      if (datasetsValue && datasetsValue.datasets.length > 0) {
        const firstDataset = datasetsValue.datasets[0];
        this.selectedDataset.set(firstDataset);
      }
    });
  }

  onDatasetChange(index: number) {
    const datasetsValue = this.datasets.value();
    if (datasetsValue) {
      this.selectedDataset.set(datasetsValue.datasets[index]);
    }
  }
}
