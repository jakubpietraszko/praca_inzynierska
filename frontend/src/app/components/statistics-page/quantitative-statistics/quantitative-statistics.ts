import { Component, signal } from '@angular/core';
import { GenericSelector } from '../../shared/generic-selector/generic-selector';

@Component({
  selector: 'app-quantitative-statistics',
  imports: [GenericSelector],
  templateUrl: './quantitative-statistics.html',
  styleUrl: './quantitative-statistics.css'
})
export class QuantitativeStatistics {
  selectedModel = signal<string | null>(null);
  selectedDataset = signal<string | null>(null);
}
