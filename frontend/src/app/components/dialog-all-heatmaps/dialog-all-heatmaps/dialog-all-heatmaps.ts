import { Component, inject, Signal, signal, WritableSignal } from '@angular/core';
import { ModelSelector } from '../../model-selector/model-selector/model-selector';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { DialogUnsupervisedActivationsData } from '../../../interfaces/dialog-unsupervised-activations-data';
import { CompareHeatmaps } from "../../compare-heatmaps/compare-heatmaps/compare-heatmaps";

@Component({
  selector: 'app-dialog-all-heatmaps',
  imports: [ModelSelector, CompareHeatmaps],
  templateUrl: './dialog-all-heatmaps.html',
  styleUrl: './dialog-all-heatmaps.css'
})
export class DialogAllHeatmaps {
  selectedModel: WritableSignal<string | null> = signal(null);

  dialogRef = inject(MatDialogRef<DialogAllHeatmaps>);
  data = inject<DialogUnsupervisedActivationsData>(MAT_DIALOG_DATA);

}
