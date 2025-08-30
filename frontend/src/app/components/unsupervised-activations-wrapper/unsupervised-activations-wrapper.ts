import {
  Component,
  computed,
  effect,
  inject,
  OnDestroy,
  OnInit,
  Signal,
  WritableSignal,
} from '@angular/core';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { signal } from '@angular/core';
import { ModelsService } from '../../services/models/models-service';
import { Subscription } from 'rxjs';
import { AllModelsResponse } from '../../interfaces/all-models';
import { DatasetsService } from '../../services/datasets/datasets-service';
import { AllDatasetsResponse } from '../../interfaces/all-datasets';
import { LabelsService } from '../../services/labels/labels-service';
import { ImagesService } from '../../services/images/images-service';
import {
  AllLabelsRequest,
  AllLabelsResponse,
} from '../../interfaces/all-labels';
import { AllImagesResponse } from '../../interfaces/all-images';
import { UnsupervisedActivations } from '../unsupervised-activations/unsupervised-activations';

@Component({
  selector: 'app-unsupervised-activations-wrapper',
  imports: [
    MatCheckboxModule,
    FormsModule,
    ReactiveFormsModule,
    MatFormFieldModule,
    MatSelectModule,
    MatInputModule,
    UnsupervisedActivations,
  ],
  templateUrl: './unsupervised-activations-wrapper.html',
  styleUrl: './unsupervised-activations-wrapper.css',
})
export class UnsupervisedActivationsWrapper implements OnInit, OnDestroy {
  modelsService: ModelsService = inject(ModelsService);
  datasetService: DatasetsService = inject(DatasetsService);
  labelsService: LabelsService = inject(LabelsService);
  imagesService: ImagesService = inject(ImagesService);

  allModels = signal<string[]>([]);
  allDatasets = signal<string[]>([]);
  allLabels = signal<string[]>([]);
  allImages = signal<string[]>([]);

  subscription = new Subscription();

  isModelsLoaded = signal(false);
  modelSelected: WritableSignal<string | null> = signal(null);

  isDatasetsLoaded = signal(false);
  datasetSelected: WritableSignal<string | null> = signal(null);

  modalAndDatasetSelected: Signal<boolean> = computed(
    () => this.modelSelected() !== null && this.datasetSelected() !== null
  );

  isLabelsLoaded = signal(false);
  labelSelected: WritableSignal<string | null> = signal(null);

  isImagesLoaded = signal(false);
  imageSelected: WritableSignal<string | null> = signal(null);

  modelDatasetLabelImageSelected: Signal<boolean> = computed(
    () =>
      this.modelSelected() !== null &&
      this.datasetSelected() !== null &&
      this.labelSelected() !== null &&
      this.imageSelected() !== null
  );

  constructor() {
    effect(() => {
      if (this.modalAndDatasetSelected()) {
        const request: AllLabelsRequest = {
          dataset: this.datasetSelected()!,
        };
        this.subscription.add(
          this.labelsService
            .allLabels(request)
            .subscribe((labels: AllLabelsResponse) => {
              this.allLabels.set(labels.labels);
              this.isLabelsLoaded.set(true);
            })
        );
      }
    });

    effect(() => {
      if (this.modalAndDatasetSelected() && this.labelSelected() !== null) {
        const request = {
          dataset: this.datasetSelected()!,
          label: this.labelSelected()!,
        };

        this.subscription.add(
          this.imagesService
            .allImages(request)
            .subscribe((response: AllImagesResponse) => {
              this.allImages.set(response.images);
              this.isImagesLoaded.set(true);
            })
        );
      }
    });
  }

  ngOnInit(): void {
    this.subscription.add(
      this.modelsService.allModels().subscribe((models: AllModelsResponse) => {
        this.allModels.set(models.models);
        this.isModelsLoaded.set(true);
      })
    );

    this.subscription.add(
      this.datasetService
        .allDatasets()
        .subscribe((datasets: AllDatasetsResponse) => {
          this.allDatasets.set(datasets.datasets);
          this.isDatasetsLoaded.set(true);
        })
    );
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }

  onModelSelected(model: string | null) {
    this.modelSelected.set(model);
  }

  onDatasetSelected(dataset: string | null) {
    this.datasetSelected.set(dataset);
    this.isLabelsLoaded.set(false);
    this.isImagesLoaded.set(false);
    this.labelSelected.set(null);
    this.imageSelected.set(null);
  }

  onLabelSelected(label: string | null) {
    this.labelSelected.set(label);
    this.isImagesLoaded.set(false);
    this.imageSelected.set(null);
  }

  onImageSelected(image: string | null) {
    this.imageSelected.set(image);
  }
}
