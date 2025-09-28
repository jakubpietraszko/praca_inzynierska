import { Component, inject, signal, OnInit, computed } from '@angular/core';
import { HeatmapsSelector } from '../../heatmaps-selector/heatmaps-selector/heatmaps-selector';
import { ModelSelector } from '../../model-selector/model-selector/model-selector';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { IDialogUnsupervisedHeatmaps } from '../../../interfaces/dialog-unsupervised-heatmaps';

@Component({
  selector: 'app-dialog-unsupervised-heatmaps',
  imports: [ModelSelector, HeatmapsSelector,],
  templateUrl: './dialog-unsupervised-heatmaps.html',
  styleUrl: './dialog-unsupervised-heatmaps.css'
})
export class DialogUnsupervisedHeatmaps implements OnInit {
  selectedModel = signal<string | null>(null);
  selectedHeatmap = signal<string | null>(null);
  dataset = signal<string | null>(null);
  label = signal<string | null>(null);

  dialogRef = inject(MatDialogRef<DialogUnsupervisedHeatmaps>);
  data = inject<IDialogUnsupervisedHeatmaps>(MAT_DIALOG_DATA);

  isAllDataSelected = computed(
    () => {
      return this.selectedModel() !== null && this.selectedHeatmap() !== null && this.dataset() !== null && this.label() !== null;
    }
  )

  ngOnInit(): void {
    this.dataset.set(this.data.dataset);
    this.label.set(this.data.label);
  }
}
