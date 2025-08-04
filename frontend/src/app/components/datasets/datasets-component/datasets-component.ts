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

@Component({
  selector: 'app-datasets-component',
  imports: [CommonModule, MatToolbar, MatButtonModule],
  templateUrl: './datasets-component.html',
  styleUrl: './datasets-component.css'
})
export class DatasetsComponent implements OnInit, OnDestroy{
  datasetsService = inject(DatasetsService);
  labelsService = inject(LabelsService);
  subscription: Subscription = new Subscription();
  labelsSubscription: Subscription = new Subscription();

  datasets: string[] | null = null;
  labels: string[] | null = null;

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
  }

  selectDataset(dataset: string): void {
    this.selectedDataset = dataset;
    this.selectedLabel = null;

    if(this.labelsSubscription) {
      this.labelsSubscription.unsubscribe();
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
  }
}
