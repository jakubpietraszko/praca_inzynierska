import { httpResource } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  effect,
  model,
} from '@angular/core';
import { AllDatasetsResponse } from '../../../interfaces/all-datasets';
import { environment } from '../../../../environments/environment';
import { MatTabsModule } from '@angular/material/tabs';

@Component({
  selector: 'app-generic-dataset-selector',
  imports: [MatTabsModule],
  templateUrl: './generic-dataset-selector.html',
  styleUrl: './generic-dataset-selector.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericDatasetSelector {
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
