import { Component, inject } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { GeneratorUnsupervisedLabel } from '../generator-unsupervised-label';
import { ContentUnsupervisedLabel } from '../content-unsupervised-label/content-unsupervised-label';

@Component({
  selector: 'app-wrapper-unsupervised-label',
  imports: [ContentUnsupervisedLabel],
  templateUrl: './wrapper-unsupervised-label.html',
  styleUrl: './wrapper-unsupervised-label.css',
  providers: [GeneratorUnsupervisedLabel],
})
export class WrapperUnsupervisedLabel {
  private readonly data = inject(MAT_DIALOG_DATA) as {
    dataset: string;
    label: string;
  };

  protected readonly unsupervisedLabel = inject(GeneratorUnsupervisedLabel);
  protected readonly datasetName = this.data.dataset;
  protected readonly labelName = this.data.label;

  constructor() {
    this.unsupervisedLabel.fetchUnsupervisedLabel({
      dataset: this.data.dataset,
      label: this.data.label,
    });
  }
}
