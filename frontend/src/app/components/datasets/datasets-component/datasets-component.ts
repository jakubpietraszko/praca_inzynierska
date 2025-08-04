import { Component, OnDestroy, OnInit } from '@angular/core';
import { DatasetsService } from '../../../services/datasets/datasets-service';
import { inject } from '@angular/core';
import { Observable } from 'rxjs';
import { CommonModule } from '@angular/common';
import { DatasetsListInterface } from '../../../interfaces/datasets-list-interface';
import { MatToolbar } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-datasets-component',
  imports: [CommonModule, MatToolbar, MatButtonModule],
  templateUrl: './datasets-component.html',
  styleUrl: './datasets-component.css'
})
export class DatasetsComponent {
  datasetsService = inject(DatasetsService);
  datasets$: Observable<DatasetsListInterface>;
  selectedDataset: string | null = null;

  constructor() {
    this.datasets$ = this.datasetsService.allDatasets()
  }

  selectDataset(dataset: string): void {
    this.selectedDataset = dataset;
  }
}
