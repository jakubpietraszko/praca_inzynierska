import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  OnInit,
  signal,
} from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ContentUnsupervisedAnyActivations } from '../content-unsupervised-any-activations/content-unsupervised-any-activations';
import { GenericSelector } from '../../shared/generic-selector/generic-selector';
import { A11yModule } from '@angular/cdk/a11y';
import { Switcher } from '../../shared/switcher/switcher';
import {
  AllLayersResponse,
  LayerConvRef,
} from '../../../interfaces/model-layers';

@Component({
  selector: 'app-wrapper-unsupervised-any-activations',
  imports: [ContentUnsupervisedAnyActivations, GenericSelector, Switcher],
  templateUrl: './wrapper-unsupervised-any-activations.html',
  styleUrl: './wrapper-unsupervised-any-activations.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WrapperUnsupervisedAnyActivations {
  data = inject(MAT_DIALOG_DATA) as {
    model: string;
    ref: string;
    layers: AllLayersResponse | null;
  };

  selectedRef = signal<string>(this.data.ref);
  items = computed(() => {
    if (!this.data.layers) {
      return [];
    }
    const all_convs = this.data.layers.layers.filter((layer: LayerConvRef) => {
      return layer.conv;
    });
    return all_convs.map((layer: LayerConvRef) => layer.ref);
  });

  selectedDataset = signal<string | null>(null);
  selectedLabel = signal<string | null>(null);
  selectedImage = signal<string | null>(null);

  allNotNull = computed(() => {
    return (
      this.selectedDataset() !== null &&
      this.selectedLabel() !== null &&
      this.selectedImage() !== null
    );
  });
}
