import { httpResource } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  effect,
  model,
} from '@angular/core';
import { AllModelsResponse } from '../../../interfaces/all-models';
import { environment } from '../../../../environments/environment';
import { MatTabsModule } from '@angular/material/tabs';

@Component({
  selector: 'app-generic-model-selector',
  imports: [MatTabsModule],
  templateUrl: './generic-model-selector.html',
  styleUrl: './generic-model-selector.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericModelSelector {
  selectedModel = model<string | null>(null);

  models = httpResource<AllModelsResponse>(
    () => `${environment.apiUrl}/all-models`,
  );

  constructor() {
    effect(() => {
      const modelsValue = this.models.value();
      if (modelsValue && modelsValue.models.length > 0) {
        const firstModel = modelsValue.models[0];
        this.selectedModel.set(firstModel);
      }
    });
  }

  onModelChange(index: number) {
    const modelsValue = this.models.value();
    if (modelsValue) {
      this.selectedModel.set(modelsValue.models[index]);
    }
  }
}
