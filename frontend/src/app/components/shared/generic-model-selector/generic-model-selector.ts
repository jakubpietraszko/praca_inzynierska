import {
  ChangeDetectionStrategy,
  Component,
  effect,
  model,
  inject,
} from '@angular/core';
import { MatTabsModule } from '@angular/material/tabs';
import { AllModels } from '../../../services/all-models';

@Component({
  selector: 'app-generic-model-selector',
  imports: [MatTabsModule],
  templateUrl: './generic-model-selector.html',
  styleUrl: './generic-model-selector.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericModelSelector {
  private readonly allModels = inject(AllModels);

  selectedModel = model<string | null>(null);

  models = this.allModels.models;

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
