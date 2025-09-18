import { Component, computed, inject, model, output } from '@angular/core';
import { ModelsService } from '../../services/models/models-service';
import { FormsModule } from '@angular/forms';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { DatasetsService } from '../../services/datasets/datasets-service';
import { HeatmapsService } from '../../services/heatmaps/heatmaps';

@Component({
  selector: 'app-model-dataset-selector',
  imports: [FormsModule, MatInputModule, MatSelectModule, MatFormFieldModule],
  templateUrl: './model-dataset-selector.html',
  styleUrl: './model-dataset-selector.css',
})
export class ModelDatasetSelector {
  modelsService = inject(ModelsService);
  datasetsService = inject(DatasetsService);
  heatmapsService = inject(HeatmapsService);

  selectedModel = model<string>();
  selectedDataset = model<string>();

  models = computed(() => this.modelsService.models()?.models || []);
  datasets = computed(() => this.datasetsService.datasets()?.datasets || []);
}
