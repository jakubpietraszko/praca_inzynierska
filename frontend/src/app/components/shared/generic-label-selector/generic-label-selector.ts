import { httpResource } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  input,
  model,
  effect,
} from '@angular/core';
import { environment } from '../../../../environments/environment';
import { CommonModule } from '@angular/common';
import { MatTabsModule } from '@angular/material/tabs';
import { AllLabelsResponse } from '../../../interfaces/all-labels';
import { ContentChild, TemplateRef } from '@angular/core';

@Component({
  selector: 'app-generic-label-selector',
  imports: [CommonModule, MatTabsModule],
  templateUrl: './generic-label-selector.html',
  styleUrl: './generic-label-selector.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericLabelSelector {
  @ContentChild('postfixButton', { static: false, read: TemplateRef })
  postfixButtonTemplate!: TemplateRef<any>;

  dataset = input.required<string>();
  selectedLabel = model<string | null>(null);

  labels = httpResource<AllLabelsResponse>(
    () => `${environment.apiUrl}/all-labels/${this.dataset()}`,
  );

  constructor() {
    effect(() => {
      const labelsValue = this.labels.value();
      if (labelsValue && labelsValue.labels.length > 0) {
        const firstLabel = labelsValue.labels[0];
        this.selectedLabel.set(firstLabel);
      }
    });
  }

  onLabelTabChange(index: number) {
    const labelsValue = this.labels.value();
    if (labelsValue) {
      this.selectedLabel.set(labelsValue.labels[index]);
    }
  }
}
