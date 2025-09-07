import { Component, OnDestroy, OnInit, resolveForwardRef, signal, computed } from '@angular/core';
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
import { MatIconModule } from '@angular/material/icon';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { DialogUnsupervisedActivations } from '../../dialog-unsupervised-activations/dialog-unsupervised-activations';
import { DialogUnsupervisedActivationsData } from '../../../interfaces/dialog-unsupervised-activations-data';
import { DialogAllHeatmaps } from '../../dialog-all-heatmaps/dialog-all-heatmaps/dialog-all-heatmaps';

@Component({
  selector: 'app-datasets-component',
  imports: [
    CommonModule,
    MatTabsModule,
    MatButtonModule,
    MatPaginatorModule,
    MatGridListModule,
    LoadImageDirective,
    MatIconModule,
    MatDialogModule,
  ],
  templateUrl: './datasets-component.html',
  styleUrl: './datasets-component.css',
})
export class DatasetsComponent implements OnInit, OnDestroy {
  datasetsService = inject(DatasetsService);
  labelsService = inject(LabelsService);
  imagesService = inject(ImagesService);
  dialog = inject(MatDialog);

  subscription: Subscription = new Subscription();
  labelsSubscription: Subscription = new Subscription();
  pathsImagesSubscription: Subscription = new Subscription();

  // Signal-based properties
  datasets = signal<string[] | null>(null);
  labels = signal<string[] | null>(null);
  pathsImages = signal<string[] | null>(null);
  selectedDataset = signal<string | null>(null);
  selectedLabel = signal<string | null>(null);
  selectedDatasetIndex = signal<number>(0);
  selectedLabelIndex = signal<number>(0);
  currentPage = signal<number>(0);
  totalImages = signal<number>(0);

  // Computed properties
  paginatedImages = computed(() => {
    const images = this.pathsImages();
    const page = this.currentPage();
    if (!images) {
      return [];
    }
    const startIndex = page * this.imagesPerPage;
    const endIndex = startIndex + this.imagesPerPage;
    return images.slice(startIndex, endIndex);
  });

  imagesPerPage = environment.imagesPerPage;
  imagesPerRow = environment.imagesPerRow;

  constructor() {}

  ngOnInit(): void {
    this.subscription.add(
      this.datasetsService
        .allDatasets()
        .subscribe((response: AllDatasetsResponse) => {
          this.datasets.set(response.datasets);
          const datasets = this.datasets();
          if (datasets && datasets.length > 0) {
            this.selectDataset(datasets[0]);
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
    this.selectedDataset.set(dataset);
    this.selectedLabel.set(null);
    this.pathsImages.set(null);
    this.selectedLabelIndex.set(0);
    this.currentPage.set(0);
    this.totalImages.set(0);

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
        this.labels.set(response.labels);
        const labels = this.labels();
        if (labels && labels.length > 0) {
          this.selectLabel(labels[0]);
        }
      });
  }

  onDatasetTabChange(index: number): void {
    const datasets = this.datasets();
    if (datasets && datasets[index]) {
      this.selectedDatasetIndex.set(index);
      this.selectDataset(datasets[index]);
    }
  }

  selectLabel(label: string): void {
    this.selectedLabel.set(label);
    
    this.pathsImages.set(null);
    this.totalImages.set(0);
    this.currentPage.set(0);

    if (this.pathsImagesSubscription) {
      this.pathsImagesSubscription.unsubscribe();
    }

    const selectedDataset = this.selectedDataset();
    if (selectedDataset) {
      const request: AllImagesRequest = {
        dataset: selectedDataset,
        label: label,
      };
      this.pathsImagesSubscription = this.imagesService
        .allImages(request)
        .subscribe((response: AllImagesResponse) => {
          this.pathsImages.set(response.images);
          this.totalImages.set(response.images.length);
          this.currentPage.set(0);
        });
    }
  }

  onLabelTabChange(index: number): void {
    const labels = this.labels();
    if (labels && labels[index]) {
      this.selectedLabelIndex.set(index);
      this.selectLabel(labels[index]);
    }
  }

  onPageChange(event: PageEvent): void {
    this.currentPage.set(event.pageIndex);
  }

  getImageFileName(path: string): string {
    return path.split('/').pop() || path;
  }

  openImageDialog(dataset: string, label: string, filename: string): void {
    const dialogData: DialogUnsupervisedActivationsData = {
      dataset: dataset,
      label: label,
      filename: filename,
    };

    this.dialog.open(DialogUnsupervisedActivations, {
      width: '90vw',
      height: '90vh',
      maxWidth: '90vw',
      maxHeight: '90vh',
      panelClass: 'large-dialog',
      hasBackdrop: true,
      disableClose: false,
      data: dialogData,
    });
  }

  openAllHeatmapsDialog(
    dataset: string,
    label: string,
    filename: string
  ): void {
    const dialogData: DialogUnsupervisedActivationsData = {
      dataset: dataset,
      label: label,
      filename: filename,
    };

    this.dialog.open(DialogAllHeatmaps, {
      width: '90vw',
      height: '90vh',
      maxWidth: '90vw',
      maxHeight: '90vh',
      panelClass: 'large-dialog',
      hasBackdrop: true,
      disableClose: false,
      data: dialogData,
    });
  }
}
