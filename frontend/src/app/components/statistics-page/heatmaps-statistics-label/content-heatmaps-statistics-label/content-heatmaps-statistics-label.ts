import { Component, input } from '@angular/core';

@Component({
  selector: 'app-content-heatmaps-statistics-label',
  imports: [],
  templateUrl: './content-heatmaps-statistics-label.html',
  styleUrl: './content-heatmaps-statistics-label.css',
})
export class ContentHeatmapsStatisticsLabel {
  model = input.required<string>();
  dataset = input.required<string>();
  heatmap = input.required<string>();
  label = input.required<string>();
}
