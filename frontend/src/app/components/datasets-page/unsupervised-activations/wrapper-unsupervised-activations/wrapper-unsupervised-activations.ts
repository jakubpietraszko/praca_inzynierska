import { Component, computed, inject, signal } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { GenericSelector } from '../../../shared/generic-selector/generic-selector';
import { ContentUnsupervisedActivations } from '../content-unsupervised-activations/content-unsupervised-activations';

@Component({
  selector: 'app-wrapper-unsupervised-activations',
  imports: [GenericSelector, ContentUnsupervisedActivations],
  templateUrl: './wrapper-unsupervised-activations.html',
  styleUrl: './wrapper-unsupervised-activations.css',
})
export class WrapperUnsupervisedActivations {
  data = inject(MAT_DIALOG_DATA) as {
    dataset: string;
    label: string;
    filename: string;
  };

  selectedModel = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);
  selectedLabel = signal<string | null>(null);
  selectedFilename = signal<string | null>(null);

  allNotNull = computed(() => {
    return (
      this.selectedModel() !== null &&
      this.selectedDataset() !== null &&
      this.selectedLabel() !== null &&
      this.selectedFilename() !== null
    );
  });

  constructor() {
    if (this.data) {
      this.selectedDataset.set(this.data.dataset);
      this.selectedLabel.set(this.data.label);
      this.selectedFilename.set(this.data.filename);
    }
  }
}
