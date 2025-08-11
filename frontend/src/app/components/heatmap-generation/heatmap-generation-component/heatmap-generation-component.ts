import { Component, inject, OnDestroy, OnInit } from '@angular/core';
import { CurrentStateService } from '../../../services/current-state/current-state-service';
import { Subscription } from 'rxjs';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-heatmap-generation-component',
  imports: [MatCardModule, MatButtonModule, MatIconModule, RouterModule, CommonModule],
  templateUrl: './heatmap-generation-component.html',
  styleUrl: './heatmap-generation-component.css'
})
export class HeatmapGenerationComponent implements OnInit, OnDestroy {
  currentStateService = inject(CurrentStateService);
  subscription = new Subscription();

  selectedModel: string | null = null;
  selectedDataset: string | null = null;

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
  }

  get isConfigurationComplete(): boolean {
    return this.selectedModel !== null && this.selectedDataset !== null;
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }
}
