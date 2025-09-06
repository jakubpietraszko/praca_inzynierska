import { Component, inject, model, OnInit, signal } from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatInputModule} from '@angular/material/input';
import {MatSelectModule} from '@angular/material/select';
import {MatFormFieldModule} from '@angular/material/form-field';
import { ModelsService } from '../../../services/models/models-service';
import { AllModelsResponse } from '../../../interfaces/all-models';

@Component({
  selector: 'app-model-selector',
  imports: [FormsModule, MatInputModule, MatSelectModule, MatFormFieldModule],
  templateUrl: './model-selector.html',
  styleUrls: ['./model-selector.css']
})
export class ModelSelector implements OnInit{

  modelsService = inject(ModelsService);
  models = signal<string[]>([]);
  selectedModel = model<string | null>(null);

  ngOnInit(): void {
    this.modelsService.allModels().subscribe(
      (response: AllModelsResponse) => {
        this.models.set(response.models);
      }
    )
  }







}
