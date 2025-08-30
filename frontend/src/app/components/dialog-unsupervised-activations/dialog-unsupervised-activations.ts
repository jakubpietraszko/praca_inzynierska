import {
  Component,
  inject,
  OnInit,
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
import { Subscription } from 'rxjs';
import { ModelsService } from '../../services/models/models-service';
import { AllModelsResponse } from '../../interfaces/all-models';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { UnsupervisedActivations } from '../unsupervised-activations/unsupervised-activations';

@Component({
  selector: 'app-dialog-unsupervised-activations',
  imports: [
    MatDialogModule,
    MatButtonModule,
    MatCheckboxModule,
    MatSelectModule,
    MatInputModule,
    FormsModule,
    ReactiveFormsModule,
    MatFormFieldModule,
    UnsupervisedActivations
  ],
  templateUrl: './dialog-unsupervised-activations.html',
  styleUrl: './dialog-unsupervised-activations.css',
})
export class DialogUnsupervisedActivations implements OnInit {
  dialogRef = inject(MatDialogRef<DialogUnsupervisedActivations>);
  data = inject<DialogUnsupervisedActivationsData>(MAT_DIALOG_DATA);

  modelsService = inject(ModelsService);

  subscription: Subscription = new Subscription();

  selectedModel: WritableSignal<string | null> = signal(null);
  allModels: WritableSignal<string[]> = signal([]);
  isModelsLoaded: WritableSignal<boolean> = signal(false);
  modelSelected: WritableSignal<string | null> = signal(null);

  ngOnInit(): void {
    this.subscription.add(
      this.modelsService
        .allModels()
        .subscribe((response: AllModelsResponse) => {
          this.allModels.set(response.models);
          this.isModelsLoaded.set(true);
        })
    );
  }

  onModelSelected(model: string | null): void {
    this.modelSelected.set(model);
  }
}
