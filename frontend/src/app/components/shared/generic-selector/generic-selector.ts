import {
  ChangeDetectionStrategy,
  Component,
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

@Component({
  selector: 'app-generic-selector',
  imports: [FormsModule, MatInputModule, MatSelectModule, MatFormFieldModule],
  templateUrl: './generic-selector.html',
  styleUrl: './generic-selector.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GenericSelector {
  model = input<boolean>(false);
  dataset = input<boolean>(false);
  heatmap = input<boolean>(false);

  selectedModel = model<string | null>();
  selectedDataset = model<string | null>();
  selectedHeatmap = model<string | null>();

  models = httpResource<AllModelsResponse>(
    () => this.model() ? `${environment.apiUrl}/all-models` : undefined
  );

  datasets = httpResource<AllDatasetsResponse>(
    () => this.dataset() ? `${environment.apiUrl}/all-datasets` : undefined
  );

  heatmaps = httpResource<AllHeatmapsResponse>(
    () => this.heatmap() ? `${environment.apiUrl}/all-heatmaps` : undefined
  );
}
