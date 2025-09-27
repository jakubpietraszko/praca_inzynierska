import { Component, input } from '@angular/core';

@Component({
  selector: 'app-content-unsupervised-heatmaps',
  imports: [],
  templateUrl: './content-unsupervised-heatmaps.html',
  styleUrl: './content-unsupervised-heatmaps.css',
})
export class ContentUnsupervisedHeatmaps {
  model = input.required<string>();
  heatmap = input.required<string>();
  dataset = input.required<string>();
  label = input.required<string>();
}
