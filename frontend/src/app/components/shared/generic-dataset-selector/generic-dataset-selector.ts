import { httpResource } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  contentChild,
  effect,
  model,
  TemplateRef,
} from '@angular/core';
import { environment } from '../../../../environments/environment';
import { MatTabsModule } from '@angular/material/tabs';
import { CommonModule } from '@angular/common';
import { AllDatasetsResponse } from '../../../generated-interfaces/all_datasets';

@Component({
  selector: 'app-generic-dataset-selector',
  imports: [MatTabsModule, CommonModule],
  templateUrl: './generic-dataset-selector.html',
  styleUrl: './generic-dataset-selector.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericDatasetSelector {
  postfixButtonTemplate = contentChild('postfixButton', { read: TemplateRef });

  selectedDataset = model<string | null>(null);

  datasets = httpResource<AllDatasetsResponse>(
    () => `${environment.apiUrl}/all-datasets`,
  );

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
