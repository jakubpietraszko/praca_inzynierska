import { Component, computed, inject, signal } from '@angular/core';
import { GenericSelector } from '../../../shared/generic-selector/generic-selector';
import { ContentUnsupervisedHeatmaps } from '../content-unsupervised-heatmaps/content-unsupervised-heatmaps';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';

@Component({
  selector: 'app-wrapper-unsupervised-heatmaps',
  imports: [GenericSelector, ContentUnsupervisedHeatmaps],
  templateUrl: './wrapper-unsupervised-heatmaps.html',
  styleUrl: './wrapper-unsupervised-heatmaps.css',
})
export class WrapperUnsupervisedHeatmaps {
  data = inject(MAT_DIALOG_DATA) as { dataset: string; label: string };

  selectedModel = signal<string | null>(null);
  selectedHeatmap = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);
  selectedLabel = signal<string | null>(null);

  allNotNull = computed(() => {
    return (
      this.selectedModel() !== null &&
      this.selectedHeatmap() !== null &&
      this.selectedDataset() !== null &&
      this.selectedLabel() !== null
    );
  });

  constructor() {
    if (this.data) {
      this.selectedDataset.set(this.data.dataset);
      this.selectedLabel.set(this.data.label);
    }
  }
}
