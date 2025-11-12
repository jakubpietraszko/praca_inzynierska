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
import { GeneratorDirectory } from './generator-directory';
import { ModelDatasetI } from '../../generated-interfaces/numerator_denominator';

@Component({
  selector: 'app-generator',
  imports: [GenericSelector, GenericLabelSelector, GeneratorList],
  templateUrl: './generator.html',
  styleUrl: './generator.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [GeneratorDirectory],
})
export class Generator {
  private directoryGenerator = inject(GeneratorDirectory);

  protected directoryCreated = this.directoryGenerator.directoryCreated;

  selectedModel = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);
  selectedLabel = signal<string | null>(null);

  showLabelSelector = computed(
    () =>
      !!this.selectedModel() &&
      !!this.selectedDataset() &&
      this.directoryCreated(),
  );

  showGeneratorList = computed(
    () =>
      !!this.selectedModel() &&
      !!this.selectedDataset() &&
      !!this.selectedLabel() &&
      this.directoryCreated(),
  );

  constructor() {
    effect(() => {
      const model = this.selectedModel();
      const dataset = this.selectedDataset();

      if (model && dataset) {
        const request: ModelDatasetI = {
          model: model,
          dataset: dataset,
        };
        this.directoryGenerator.createDirectory(request);
      }
    });
  }
}
