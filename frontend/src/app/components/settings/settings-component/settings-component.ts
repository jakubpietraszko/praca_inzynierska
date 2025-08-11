import { Component, inject, OnDestroy, OnInit } from '@angular/core';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { FormsModule } from '@angular/forms';
import { ModelsService } from '../../../services/models/models-service';
import { DatasetsService } from '../../../services/datasets/datasets-service';
import { CurrentStateService } from '../../../services/current-state/current-state-service';
import { Subscription } from 'rxjs';
import { AllModelsResponse } from '../../../interfaces/all-models';
import { AllDatasetsResponse } from '../../../interfaces/all-datasets';

@Component({
  selector: 'app-settings-component',
  imports: [MatFormFieldModule, MatSelectModule, MatInputModule, FormsModule],
  templateUrl: './settings-component.html',
  styleUrl: './settings-component.css',
})
export class SettingsComponent implements OnInit, OnDestroy{
  modelsService: ModelsService = inject(ModelsService);
  datasetsService: DatasetsService = inject(DatasetsService);
  currentStateService: CurrentStateService = inject(CurrentStateService);
  subscription: Subscription = new Subscription();

  models: string[] | null = null;
  datasets: string[] | null = null

  selectedModel: string | null = null;
  selectedDataset: string | null = null;

  modelOptions: { label: string, value: string | null }[] = [
    { label: 'Nie wybrano', value: null }
  ];
  
  datasetOptions: { label: string, value: string | null }[] = [
    { label: 'Nie wybrano', value: null }
  ];

  ngOnInit(): void {
    this.subscription.add(
      this.currentStateService.model$.subscribe(model => {
        this.selectedModel = model;
      })
    );

    this.subscription.add(
      this.currentStateService.dataset$.subscribe(dataset => {
        this.selectedDataset = dataset;
      })
    );

    this.subscription.add(
      this.modelsService.allModels()
        .subscribe(
          (response: AllModelsResponse) => {
            this.models = response.models;
            this.modelOptions = [
              { label: 'Nie wybrano', value: null },
              ...response.models.map(model => ({ label: model, value: model }))
            ];
          }
        )
    );

    this.subscription.add(
      this.datasetsService.allDatasets()
        .subscribe(
          (response: AllDatasetsResponse) => {
            this.datasets = response.datasets;
            this.datasetOptions = [
              { label: 'Nie wybrano', value: null },
              ...response.datasets.map(dataset => ({ label: dataset, value: dataset }))
            ];
          }
        )
    );
  }

  onModelChange(): void {
    this.currentStateService.setModel(this.selectedModel);
  }

  onDatasetChange(): void {
    this.currentStateService.setDataset(this.selectedDataset);
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe(); 
  }
}
