import { Component, OnDestroy, OnInit, resolveForwardRef } from '@angular/core';
import { DatasetsService } from '../../../services/datasets/datasets-service';
import { inject } from '@angular/core';
import { Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
import { MatTabsModule } from '@angular/material/tabs';
import { MatButtonModule } from '@angular/material/button';
import { MatPaginatorModule, PageEvent } from '@angular/material/paginator';
import { MatGridListModule } from '@angular/material/grid-list';
import { LabelsService } from '../../../services/labels/labels-service';
import { ImagesService } from '../../../services/images/images-service';
import { AllDatasetsResponse } from '../../../interfaces/all-datasets';
import {
  AllLabelsRequest,
  AllLabelsResponse,
} from '../../../interfaces/all-labels';
import {
  AllImagesRequest,
  AllImagesResponse,
} from '../../../interfaces/all-images';
import { LoadImageDirective } from '../../../directives/load-image/load-image-directive';
import { environment } from '../../../../environments/environment';

@Component({
  selector: 'app-datasets-component',
  imports: [
    CommonModule,
    MatTabsModule,
    MatButtonModule,
    MatPaginatorModule,
    MatGridListModule,
    LoadImageDirective,
  ],
  templateUrl: './datasets-component.html',
  styleUrl: './datasets-component.css',
})
export class DatasetsComponent implements OnInit, OnDestroy {
  datasetsService = inject(DatasetsService);
  labelsService = inject(LabelsService);
  imagesService = inject(ImagesService);

  subscription: Subscription = new Subscription();
  labelsSubscription: Subscription = new Subscription();
  pathsImagesSubscription: Subscription = new Subscription();

  datasets: string[] | null = null;
  labels: string[] | null = null;
  pathsImages: string[] | null = null;
  paginatedImages: string[] = [];

  selectedDataset: string | null = null;
  selectedLabel: string | null = null;
  selectedDatasetIndex: number = 0;
  selectedLabelIndex: number = 0;

  imagesPerPage = environment.imagesPerPage;
  imagesPerRow = environment.imagesPerRow;
  currentPage = 0;
  totalImages = 0;

  constructor() {}

  ngOnInit(): void {
    this.subscription.add(
      this.datasetsService
        .allDatasets()
        .subscribe((response: AllDatasetsResponse) => {
          this.datasets = response.datasets;
          // Automatically select the first dataset on initial load
          if (this.datasets && this.datasets.length > 0) {
            this.selectDataset(this.datasets[0]);
          }
        })
    );
  }

  ngOnDestroy(): void {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
    if (this.labelsSubscription) {
      this.labelsSubscription.unsubscribe();
    }
    if (this.pathsImagesSubscription) {
      this.pathsImagesSubscription.unsubscribe();
    }
  }

  selectDataset(dataset: string): void {
    this.selectedDataset = dataset;
    this.selectedLabel = null;
    this.pathsImages = null;
    this.paginatedImages = [];
    this.selectedLabelIndex = 0;
    this.currentPage = 0;
    this.totalImages = 0;

    if (this.labelsSubscription) {
      this.labelsSubscription.unsubscribe();
    }

    if (this.pathsImagesSubscription) {
      this.pathsImagesSubscription.unsubscribe();
    }

    const request: AllLabelsRequest = {
      dataset: dataset,
    };

    this.labelsSubscription = this.labelsService
      .allLabels(request)
      .subscribe((response: AllLabelsResponse) => {
        console.log(response.labels);
        this.labels = response.labels;
        // Automatically select the first label when labels are loaded
        if (this.labels && this.labels.length > 0) {
          this.selectLabel(this.labels[0]);
        }
      });
  }

  onDatasetTabChange(index: number): void {
    if (this.datasets && this.datasets[index]) {
      this.selectedDatasetIndex = index;
      this.selectDataset(this.datasets[index]);
    }
  }

  selectLabel(label: string): void {
    this.selectedLabel = label;

    if (this.pathsImagesSubscription) {
      this.pathsImagesSubscription.unsubscribe();
    }

    if (this.selectedDataset) {
      const request: AllImagesRequest = {
        dataset: this.selectedDataset,
        label: label,
      };
      this.pathsImagesSubscription = this.imagesService
        .allImages(request)
        .subscribe((response: AllImagesResponse) => {
          this.pathsImages = response.images;
          this.totalImages = response.images.length;
          this.currentPage = 0;
          this.updatePaginatedImages();
        });
    }
  }

  onLabelTabChange(index: number): void {
    if (this.labels && this.labels[index]) {
      this.selectedLabelIndex = index;
      this.selectLabel(this.labels[index]);
    }
  }

  onPageChange(event: PageEvent): void {
    this.currentPage = event.pageIndex;
    this.updatePaginatedImages();
  }

  updatePaginatedImages(): void {
    if (!this.pathsImages) {
      this.paginatedImages = [];
      return;
    }

    const startIndex = this.currentPage * this.imagesPerPage;
    const endIndex = startIndex + this.imagesPerPage;
    this.paginatedImages = this.pathsImages.slice(startIndex, endIndex);
  }

  getImageFileName(path: string): string {
    return path.split('/').pop() || path;
  }
}
