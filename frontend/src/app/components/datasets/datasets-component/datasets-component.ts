import { Component, OnDestroy, OnInit, resolveForwardRef } from '@angular/core';
import { DatasetsService } from '../../../services/datasets/datasets-service';
import { inject } from '@angular/core';
import { Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
import { MatToolbar } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { LabelsService } from '../../../services/labels/labels-service';
import { ImagesService } from '../../../services/images/images-service';
import { AllDatasetsResponse } from '../../../interfaces/all-datasets';
import { AllLabelsRequest, AllLabelsResponse } from '../../../interfaces/all-labels';
import { AllImagesRequest, AllImagesResponse } from '../../../interfaces/all-images';

@Component({
  selector: 'app-datasets-component',
  imports: [CommonModule, MatToolbar, MatButtonModule],
  templateUrl: './datasets-component.html',
  styleUrl: './datasets-component.css'
})
export class DatasetsComponent implements OnInit, OnDestroy{
  datasetsService = inject(DatasetsService);
  labelsService = inject(LabelsService);
  imagesService = inject(ImagesService);

  subscription: Subscription = new Subscription();
  labelsSubscription: Subscription = new Subscription();
  pathsImagesSubscription: Subscription = new Subscription();

  datasets: string[] | null = null;
  labels: string[] | null = null;
  pathsImages: string[] | null = null;

  selectedDataset: string | null = null;
  selectedLabel: string | null = null;

  constructor() {

  }

  ngOnInit(): void {
    this.subscription.add(
      this.datasetsService.allDatasets()
        .subscribe(
          (response: AllDatasetsResponse) => {
            this.datasets = response.datasets;
          }
      )
    );
  }

  ngOnDestroy(): void {
    if(this.subscription) {
      this.subscription.unsubscribe();
    }
    if(this.labelsSubscription) {
      this.labelsSubscription.unsubscribe();
    }
    if(this.pathsImagesSubscription) {
      this.pathsImagesSubscription.unsubscribe();
    }
  }

  selectDataset(dataset: string): void {
    this.selectedDataset = dataset;
    this.selectedLabel = null;
    this.pathsImages = null;

    if(this.labelsSubscription) {
      this.labelsSubscription.unsubscribe();
    }

    if(this.pathsImagesSubscription) {
      this.pathsImagesSubscription.unsubscribe();
    }

    const request: AllLabelsRequest = {
      dataset: dataset
    };


    this.labelsSubscription = this.labelsService.allLabels(request)
    .subscribe((response: AllLabelsResponse) => {
      console.log(response.labels);
      this.labels = response.labels;
    });
  }

  selectLabel(label: string): void {
    this.selectedLabel = label;

    if(this.pathsImagesSubscription) {
      this.pathsImagesSubscription.unsubscribe();
    }

    if(this.selectedDataset) {
      const request: AllImagesRequest = {
        dataset: this.selectedDataset,
        label: label
      }
      this.pathsImagesSubscription = this.imagesService.allImages(request)
        .subscribe((response: AllImagesResponse) => {
          this.pathsImages = response.images;
        });
    }
  }
}
