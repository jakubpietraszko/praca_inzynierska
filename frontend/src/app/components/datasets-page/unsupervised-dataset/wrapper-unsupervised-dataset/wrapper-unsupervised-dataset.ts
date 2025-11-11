import { Component, inject } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { GeneratorUnsupervisedDataset } from '../generator-unsupervised-dataset';
import { ContentUnsupervisedDataset } from '../content-unsupervised-dataset/content-unsupervised-dataset';

@Component({
  selector: 'app-wrapper-unsupervised-dataset',
  imports: [ContentUnsupervisedDataset],
  templateUrl: './wrapper-unsupervised-dataset.html',
  styleUrl: './wrapper-unsupervised-dataset.css',
  providers: [GeneratorUnsupervisedDataset],
})
export class WrapperUnsupervisedDataset {
  private readonly data = inject(MAT_DIALOG_DATA) as {
    dataset: string;
  };

  protected readonly unsupervisedDataset = inject(GeneratorUnsupervisedDataset);
  protected readonly datasetName = this.data.dataset;

  constructor() {
    this.unsupervisedDataset.fetchUnsupervisedDataset({
      dataset: this.data.dataset,
    });
  }
}
