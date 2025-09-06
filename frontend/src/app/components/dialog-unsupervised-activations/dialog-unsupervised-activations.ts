import {
  Component,
  inject,
  signal,
  WritableSignal,
} from '@angular/core';
import {
  MAT_DIALOG_DATA,
  MatDialogRef,
  MatDialogModule,
} from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { DialogUnsupervisedActivationsData } from '../../interfaces/dialog-unsupervised-activations-data';
import { UnsupervisedActivations } from '../unsupervised-activations/unsupervised-activations';
import { ModelSelector } from '../model-selector/model-selector/model-selector';
import { CompareHeatmaps } from '../compare-heatmaps/compare-heatmaps/compare-heatmaps';
@Component({
  selector: 'app-dialog-unsupervised-activations',
  imports: [
    MatDialogModule,
    MatButtonModule,
    UnsupervisedActivations,
    ModelSelector
  ],
  templateUrl: './dialog-unsupervised-activations.html',
  styleUrl: './dialog-unsupervised-activations.css',
})
export class DialogUnsupervisedActivations {
  dialogRef = inject(MatDialogRef<DialogUnsupervisedActivations>);
  data = inject<DialogUnsupervisedActivationsData>(MAT_DIALOG_DATA);

  selectedModel: WritableSignal<string | null> = signal(null);
}

