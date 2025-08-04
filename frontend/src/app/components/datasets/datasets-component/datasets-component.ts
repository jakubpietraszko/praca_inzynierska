import { Component, OnDestroy, OnInit, resolveForwardRef } from '@angular/core';
import { DatasetsService } from '../../../services/datasets/datasets-service';
import { inject } from '@angular/core';
import { Observable, Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
import { DatasetsListInterface } from '../../../interfaces/datasets-list-interface';
import { MatToolbar } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { LabelsService } from '../../../services/labels/labels-service';
import { LabelsListInterface } from '../../../interfaces/labels-list-interface';
import { LabelsListRequestInterface } from '../../../interfaces/labels-list-request-interface';
import { PathsImagesService } from '../../../services/paths-images/paths-images-service';
import { PathsImagesRequestInterface } from '../../../interfaces/paths-images-request-interface';
import { PathsImagesListInterface } from '../../../interfaces/paths-images-list-interface';

@Component({
  selector: 'app-datasets-component',
  imports: [CommonModule, MatToolbar, MatButtonModule],
  templateUrl: './datasets-component.html',
  styleUrl: './datasets-component.css'
})
export class DatasetsComponent implements OnInit, OnDestroy{
  datasetsService = inject(DatasetsService);
  labelsService = inject(LabelsService);
  pathsImagesService = inject(PathsImagesService);

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
          (response: DatasetsListInterface) => {
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

    const request: LabelsListRequestInterface = {
      dataset: dataset
    };


    this.labelsSubscription = this.labelsService.allLabels(request)
    .subscribe((response: LabelsListInterface) => {
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
      const request: PathsImagesRequestInterface = {
        dataset: this.selectedDataset,
        label: label
      }
      this.pathsImagesSubscription = this.pathsImagesService.allPathsImages(request)
        .subscribe((response: PathsImagesListInterface) => {
          console.log(response.pathsImages);
          this.pathsImages = response.pathsImages;
        });
    }
  }
}
