import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';
import { GenericSelector } from '../shared/generic-selector/generic-selector';
import { GenericLabelSelector } from '../shared/generic-label-selector/generic-label-selector';
import { GeneratorList } from './generator-list/generator-list';
import { HeatmapsGenerator } from '../../services/heatmaps-generator';

@Component({
  selector: 'app-generator',
  imports: [GenericSelector, GenericLabelSelector, GeneratorList],
  templateUrl: './generator.html',
  styleUrl: './generator.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Generator {
  private heatmapsGenerator = inject(HeatmapsGenerator);

  selectedModel = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);
  selectedLabel = signal<string | null>(null);

  directoryCreated = signal(false);

  constructor() {
    effect(() => {
      const model = this.selectedModel();
      const dataset = this.selectedDataset();
      this.directoryCreated.set(false);

      if (model && dataset) {
        this.heatmapsGenerator
          .createDatasetsModelsDir({
            model,
            dataset,
          })
          .subscribe({
            next: () => {
              this.directoryCreated.set(true);
            },
          });
      }
    });
  }

  showLabelSelector = computed(
    () =>
      !!this.selectedModel() &&
      !!this.selectedDataset() &&
      this.directoryCreated()
  );

  showGeneratorList = computed(
    () =>
      !!this.selectedModel() &&
      !!this.selectedDataset() &&
      !!this.selectedLabel() &&
      this.directoryCreated()
  );
}
