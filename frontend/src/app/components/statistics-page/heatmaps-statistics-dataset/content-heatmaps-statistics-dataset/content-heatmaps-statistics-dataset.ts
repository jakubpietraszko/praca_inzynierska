import { ChangeDetectionStrategy, Component, input } from '@angular/core';

@Component({
  selector: 'app-content-heatmaps-statistics-dataset',
  imports: [],
  templateUrl: './content-heatmaps-statistics-dataset.html',
  styleUrl: './content-heatmaps-statistics-dataset.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ContentHeatmapsStatisticsDataset {
  model = input.required<string>();
  dataset = input.required<string>();
  heatmap = input.required<string>();
}
