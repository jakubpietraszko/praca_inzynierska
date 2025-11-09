import {
  ChangeDetectionStrategy,
  Component,
  effect,
  input,
  model,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { httpResource } from '@angular/common/http';
import { environment } from '../../../../environments/environment';
import { TranslateXaiMethodsPipe } from '../../../pipes/translate-xai-methods-pipe';
import { AllLabelsResponse } from '../../../generated-interfaces/all_labels';
import { AllImagesResponse } from '../../../generated-interfaces/all_images';
import { AllModels } from '../../../services/all-models';
import { AllDatasets } from '../../../services/all-datasets';
import { AllHeatmaps } from '../../../services/all-heatmaps';
import { inject } from '@angular/core';

@Component({
  selector: 'app-generic-selector',
  imports: [
    FormsModule,
    MatInputModule,
    MatSelectModule,
    MatFormFieldModule,
    TranslateXaiMethodsPipe,
  ],
  templateUrl: './generic-selector.html',
  styleUrl: './generic-selector.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericSelector {
  private readonly allModels = inject(AllModels);
  private readonly allDatasets = inject(AllDatasets);
  private readonly allHeatmaps = inject(AllHeatmaps);

  model = input<boolean>(false);
  dataset = input<boolean>(false);
  heatmap = input<boolean>(false);
  label = input<boolean>(false);
  image = input<boolean>(false);

  selectedModel = model<string | null>();
  selectedDataset = model<string | null>();
  selectedHeatmap = model<string | null>();
  selectedLabel = model<string | null>();
  selectedImage = model<string | null>();

  models = this.allModels.models;
  datasets = this.allDatasets.datasets;
  heatmaps = this.allHeatmaps.heatmaps;

  constructor() {
    effect(() => {
      this.selectedLabel();
      this.selectedImage.set(null);
    });

    effect(() => {
      this.selectedDataset();
      this.selectedLabel.set(null);
      this.selectedImage.set(null);
    });
  }

  labels = httpResource<AllLabelsResponse>(() => {
    const dataset = this.selectedDataset();
    if (!this.label() || !dataset) return undefined;
    return {
      url: `${environment.apiUrl}/all-labels/${dataset}`,
      method: 'GET',
    };
  });

  images = httpResource<AllImagesResponse>(() => {
    const dataset = this.selectedDataset();
    const label = this.selectedLabel();
    if (!this.image() || !dataset || !label) return undefined;
    return {
      url: `${environment.apiUrl}/all-images`,
      method: 'GET',
      params: { dataset, label },
    };
  });
}
