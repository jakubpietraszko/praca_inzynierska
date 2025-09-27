import {
  ChangeDetectionStrategy,
  Component,
  inject,
  signal,
} from '@angular/core';
import { GenericDatasetSelector } from '../shared/generic-dataset-selector/generic-dataset-selector';
import { GenericLabelSelector } from '../shared/generic-label-selector/generic-label-selector';
import { MatIconModule } from '@angular/material/icon';
import { DialogCompareHeatmaps } from './compare-heatmaps/dialog-compare-heatmaps';
import { DialogUnsupervisedHeatmaps } from './unsupervised-heatmaps/dialog-unsupervised-heatmaps';
import { DialogUnsupervisedActivations } from './unsupervised-activations/dialog-unsupervised-activations';

@Component({
  selector: 'app-datasets',
  imports: [GenericDatasetSelector, GenericLabelSelector, MatIconModule],
  templateUrl: './datasets.html',
  styleUrl: './datasets.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Datasets {
  compareHeatmapsDialog = inject(DialogCompareHeatmaps);
  unsupervisedActivationsDialog = inject(DialogUnsupervisedActivations);
  unsupervisedHeatmapsDialog = inject(DialogUnsupervisedHeatmaps);

  currentDataset = signal<string | null>(null);
  currentLabel = signal<string | null>(null);

  onUnsupervisedHeatmapOpen(dataset: string, label: string) {
    this.unsupervisedHeatmapsDialog.open(dataset, label);
  }
}
