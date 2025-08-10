import { Component, inject, OnDestroy, OnInit } from '@angular/core';
import { ModelsService } from '../../../services/models/models-service';
import { identity, Subscription } from 'rxjs';
import { AllModelsResponse } from '../../../interfaces/all-models';
import { CommonModule } from '@angular/common';
import { MatTabsModule } from '@angular/material/tabs';
import { ModelLayersService } from '../../../services/model-layers/model-layers-service';

@Component({
  selector: 'app-models-component',
  imports: [CommonModule, MatTabsModule],
  templateUrl: './models-component.html',
  styleUrl: './models-component.css',
})
export class ModelsComponent implements OnInit, OnDestroy {
  modelsService = inject(ModelsService);
  modelLayersService = inject(ModelLayersService);

  models: string[] = [];
  selectedModel: string | null = null;
  selectedModelIndex: number = 0;
  modelLayers: { [modelName: string]: string[] } = {};

  subscription: Subscription = new Subscription();
  layersSubscription: Subscription = new Subscription();

  ngOnInit(): void {
    this.subscription.add(
      this.modelsService
        .allModels()
        .subscribe((response: AllModelsResponse) => {
          this.models = response.models;
          // Fetch layers for each model
          this.models.forEach(model => {
            this.fetchModelLayers(model);
          });
        })
    );
  }

  fetchModelLayers(modelName: string): void {
    this.layersSubscription.add(
      this.modelLayersService
        .modelLayers({ model: modelName })
        .subscribe(response => {
          this.modelLayers[modelName] = response.layers;
        })
    );
  }

  ngOnDestroy(): void {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
    if (this.layersSubscription) {
      this.layersSubscription.unsubscribe();
    }
  }

  onLabelTabChange(index: number): void {
    console.log(index);
    this.selectedModelIndex = index;
  }
}
