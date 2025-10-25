import {
  ChangeDetectionStrategy,
  Component,
  inject,
  signal,
} from '@angular/core';
import { httpResource } from '@angular/common/http';
import { GenericModelSelector } from '../shared/generic-model-selector/generic-model-selector';
import { environment } from '../../../environments/environment';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { DialogUnsupervisedAnyActivations } from './dialog-unsupervised-any-activations';
import { AllLayersResponse } from '../../generated-interfaces/model_layers';

@Component({
  selector: 'app-models',
  imports: [GenericModelSelector, MatButtonModule, MatIconModule],
  templateUrl: './models.html',
  styleUrl: './models.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Models {
  currentModel = signal<string | null>(null);

  unsupervisedAnyActivationsDialog = inject(DialogUnsupervisedAnyActivations);

  layers = httpResource<AllLayersResponse>(() => {
    const model = this.currentModel();
    if (!model) return undefined;
    return {
      url: `${environment.apiUrl}/all-layers`,
      params: { model },
    };
  });

  onOpenDialog(model: string | null, ref: string | undefined) {
    if (!model || !ref) return;

    this.unsupervisedAnyActivationsDialog.open(
      model,
      ref,
      this.layers.isLoading() ? null : (this.layers.value() ?? null),
    );
  }
}
