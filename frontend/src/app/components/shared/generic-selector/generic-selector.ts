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
import { AllModelsResponse } from '../../../interfaces/all-models';
import { environment } from '../../../../environments/environment';
import { AllDatasetsResponse } from '../../../interfaces/all-datasets';
import { AllHeatmapsResponse } from '../../../interfaces/all-heatmaps';
import { TranslateXaiMethodsPipe } from '../../../pipes/translate-xai-methods-pipe';
import { AllLabelsResponse } from '../../../interfaces/all-labels';
import { AllImagesResponse } from '../../../interfaces/all-images';

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

  models = httpResource<AllModelsResponse>(() =>
    this.model() ? `${environment.apiUrl}/all-models` : undefined
  );

  datasets = httpResource<AllDatasetsResponse>(() =>
    this.dataset() ? `${environment.apiUrl}/all-datasets` : undefined
  );

  heatmaps = httpResource<AllHeatmapsResponse>(() =>
    this.heatmap() ? `${environment.apiUrl}/heatmap/all-heatmaps` : undefined
  );

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
